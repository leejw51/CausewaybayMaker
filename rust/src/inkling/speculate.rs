//! Speculative decoding for Inkling 2: the draft network proposes a few tokens, the target
//! model checks the current token and all of them in one pass, and the longest accepted run is
//! kept along with the target's own next token. A pass over several rows costs much less than
//! that many single passes, so every accepted proposal is nearly free.
//!
//! The output is the target's: greedy decoding keeps a proposal only where it equals the
//! target's choice, and sampled decoding uses rejection sampling, which draws from the target's
//! distribution exactly.
//!
//! How many tokens to propose is learned while decoding (`Controller`): the depth with the best
//! expected tokens per millisecond, from the measured cost of each depth and the measured
//! acceptance at each position. Depth zero is plain decoding.
//!
//! Ported from Ollama's MLX runner (ollama/ollama, mlxrunner/speculate.go, speculate_depth.go
//! and dflash.go, MIT License).

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::{Context, Result};
use mlx_rs::ops::indexing::{IndexOp, argmax_axis, take_along_axis};
use mlx_rs::ops::{concatenate, maximum, minimum};
use mlx_rs::{Array, Dtype, random, transforms};

use super::{Cache, Model};
use crate::llm::{Sampler, Stats};

/// Base number of rounds between probes of a deeper proposal, and the cap its backoff reaches.
const PROBE_INTERVAL: usize = 4;
const PROBE_INTERVAL_MAX: usize = 2 * PROBE_INTERVAL;
/// Weight of the newest cost sample, and how far one sample may move the average (a stall
/// must not flip the comparison against plain decoding).
const COST_ALPHA: f64 = 0.3;
const COST_CLAMP: f64 = 0.25;
/// Weight of the newest acceptance outcome: acceptance drifts with the text.
const ACCEPTANCE_ALPHA: f64 = 0.1;

/// Milliseconds per round, by proposal depth.
#[derive(Default)]
struct CostModel {
    ewma: BTreeMap<usize, f64>,
}

impl CostModel {
    fn observe(&mut self, depth: usize, ms: f64) {
        if ms <= 0.0 {
            return;
        }
        match self.ewma.get_mut(&depth) {
            Some(v) => {
                let limit = COST_CLAMP * *v;
                *v += COST_ALPHA * (ms - *v).clamp(-limit, limit);
            }
            None => {
                self.ewma.insert(depth, ms);
            }
        }
    }

    /// Two depths are needed before depths can be compared.
    fn ready(&self) -> bool {
        self.ewma.len() >= 2
    }

    fn sampled(&self, depth: usize) -> bool {
        self.ewma.contains_key(&depth)
    }

    /// Linear between the sampled depths, flat beyond them.
    fn cost(&self, depth: usize) -> f64 {
        let below = self.ewma.range(..=depth).next_back();
        let above = self.ewma.range(depth..).next();
        match (below, above) {
            (Some((&lo, &a)), Some((&hi, &b))) if hi > lo => a + (depth - lo) as f64 / (hi - lo) as f64 * (b - a),
            (Some((_, &a)), _) | (None, Some((_, &a))) => a,
            (None, None) => 0.0,
        }
    }
}

/// The chance that position i of a proposal is accepted, given that all before it were.
/// Positions past the first share one rate: they accept at similar rates, and pooling lets
/// every reached position inform the deeper ones.
#[derive(Default)]
struct AcceptanceModel {
    rate: Vec<f64>,
    seen: Vec<u32>,
    deep: f64,
    deep_seen: u32,
}

impl AcceptanceModel {
    fn observe(&mut self, drafted: usize, accepted: usize) {
        for i in 1..=drafted {
            if accepted + 1 < i {
                break; // the run ended before position i
            }
            if self.seen.len() <= i {
                self.rate.resize(i + 1, 0.0);
                self.seen.resize(i + 1, 0);
            }
            let outcome = if accepted >= i { 1.0 } else { 0.0 };
            if self.seen[i] == 0 {
                self.rate[i] = outcome;
            } else {
                self.rate[i] += ACCEPTANCE_ALPHA * (outcome - self.rate[i]);
            }
            self.seen[i] += 1;
            if i >= 2 {
                if self.deep_seen == 0 {
                    self.deep = outcome;
                } else {
                    self.deep += ACCEPTANCE_ALPHA * (outcome - self.deep);
                }
                self.deep_seen += 1;
            }
        }
    }

    fn trusted(&self, i: usize) -> bool {
        i >= 1 && self.seen.get(i).is_some_and(|&s| s >= 1)
    }

    fn acceptance(&self, i: usize) -> f64 {
        if i >= 2 && self.deep_seen >= 1 {
            return self.deep;
        }
        if self.trusted(i) {
            return self.rate[i];
        }
        // An unmeasured position inherits the deepest measured rate, or optimism.
        (1..i).rev().find(|&j| self.trusted(j)).map_or(1.0, |j| self.rate[j])
    }

    /// Expected tokens from a round at `depth`: the current token, plus each position weighted
    /// by the chance its whole prefix was accepted.
    fn expected(&self, depth: usize) -> f64 {
        let (mut total, mut run) = (1.0, 1.0);
        for i in 1..=depth {
            run *= self.acceptance(i);
            total += run;
        }
        total
    }

    /// The deepest position measured without a gap.
    fn frontier(&self) -> usize {
        (1..self.seen.len()).take_while(|&i| self.trusted(i)).last().unwrap_or(0)
    }
}

/// What the depth choice has learned, kept with the conversation.
#[derive(Default)]
pub struct Learned {
    cost: CostModel,
    acceptance: AcceptanceModel,
}

/// Chooses each round's proposal depth for one request.
struct Controller {
    /// The deepest proposal the draft network makes.
    limit: usize,
    /// INKLING_DEPTH: always this depth.
    fixed: Option<usize>,
    depth: usize,
    probed: bool,
    probe_interval: usize,
    probe_since: usize,
    last_selected: usize,
    /// For the cost samples: a round's time runs to the next round's start, and counts only
    /// when the round before it had the same depth (a change of shape inflates it).
    round_start: Option<Instant>,
    previous_depth: Option<usize>,
    round_depth: Option<usize>,
    // Statistics.
    rounds: usize,
    drafted: usize,
    accepted: usize,
}

impl Controller {
    /// A request starts with a whole block, which gives evidence for every position.
    fn new(limit: usize) -> Self {
        let fixed = std::env::var("INKLING_DEPTH").ok().and_then(|v| v.parse::<usize>().ok()).map(|d| d.min(limit));
        Self {
            limit,
            fixed,
            depth: fixed.unwrap_or(limit),
            probed: false,
            probe_interval: PROBE_INTERVAL,
            probe_since: 0,
            last_selected: 0,
            round_start: None,
            previous_depth: None,
            round_depth: None,
            rounds: 0,
            drafted: 0,
            accepted: 0,
        }
    }

    fn begin_round(&mut self, learned: &mut Learned) {
        let now = Instant::now();
        if let (Some(start), Some(depth)) = (self.round_start, self.round_depth) {
            if self.previous_depth == Some(depth) {
                learned.cost.observe(depth, now.duration_since(start).as_secs_f64() * 1e3);
            }
            self.previous_depth = Some(depth);
        }
        self.round_start = Some(now);
    }

    /// A round that is no sample of a depth's cost (it waited on earlier work).
    fn untimed_round(&mut self) {
        self.round_depth = None;
        self.rounds += 1;
    }

    /// `observed`: the leading positions that say something about acceptance (all of them,
    /// except that an accepted end-of-turn token hides the ones after it).
    fn end_round(&mut self, learned: &mut Learned, drafted: usize, accepted: usize, observed: usize) {
        self.round_depth = Some(drafted);
        self.rounds += 1;
        self.drafted += drafted;
        self.accepted += accepted;
        // A round that could not propose (no budget left) keeps its choice.
        if self.fixed.is_some() || (self.depth > 0 && drafted == 0) {
            return;
        }
        if observed > 0 {
            learned.acceptance.observe(observed, accepted);
        }
        self.depth = self.next(learned);
    }

    /// The best measured depth, with a periodic probe one deeper, after first dwelling at each
    /// depth up to one past the acceptance frontier to measure its cost.
    fn next(&mut self, learned: &Learned) -> usize {
        let limit = (learned.acceptance.frontier() + 1).min(self.limit);
        let selected = self.selected(learned, limit);
        if selected != self.last_selected {
            self.probe_interval = PROBE_INTERVAL;
            self.probe_since = 0;
            self.last_selected = selected;
        } else if self.probed {
            self.probe_interval = (self.probe_interval * 2).min(PROBE_INTERVAL_MAX);
        }
        self.probed = false;
        self.probe_since += 1;
        if self.probe_since >= self.probe_interval {
            self.probe_since = 0;
            // From plain decoding, one whole block probes every position.
            let probe = if selected == 0 { self.limit } else { (selected + 1).min(limit) };
            if probe != selected {
                self.probed = true;
                return probe;
            }
        }
        (0..=limit).find(|&d| !learned.cost.sampled(d)).unwrap_or(selected)
    }

    fn selected(&self, learned: &Learned, limit: usize) -> usize {
        if !learned.cost.ready() {
            return 0;
        }
        let value = |d: usize| learned.acceptance.expected(d) / learned.cost.cost(d);
        (0..=limit).fold(0, |best, d| if value(d) > value(best) { d } else { best })
    }

    fn report(&self, learned: &Learned, tokens: usize) {
        let costs: Vec<String> = learned.cost.ewma.iter().map(|(d, ms)| format!("{d}:{ms:.0}ms")).collect();
        eprintln!(
            "[speculation: {} rounds, {:.2} tokens/round, {}/{} proposals accepted ({:.0}%), cost by depth {}]",
            self.rounds,
            tokens as f64 / self.rounds.max(1) as f64,
            self.accepted,
            self.drafted,
            100.0 * self.accepted as f64 / self.drafted.max(1) as f64,
            costs.join(" ")
        );
    }
}

impl Learned {
    /// Start from measured round costs (milliseconds by depth), so a conversation does not
    /// spend its first rounds measuring them.
    pub(crate) fn seeded(costs: &[f64]) -> Self {
        let mut learned = Self::default();
        for (depth, &ms) in costs.iter().enumerate() {
            learned.cost.observe(depth, ms);
        }
        learned
    }
}

/// Time one round at every depth on a scratch conversation: milliseconds by depth. Run once
/// at load; it also makes Metal compile the kernels for every number of rows now, not in the
/// middle of the first answer.
pub fn calibrate(model: &Model) -> Result<Vec<f64>> {
    let draft = model.draft.as_ref().context("no draft network")?;
    let greedy = Sampler { temp: 0.0, top_k: 0 };
    let mut cache = model.make_cache();
    let warmup: Vec<i32> = (0..16).map(|i| 1000 + i).collect();
    model.forward(&Array::from_slice(&warmup, &[1, 16]), &mut cache)?.eval()?;
    let mut costs = Vec::new();
    for depth in 0..=draft.limit() {
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            if depth == 0 {
                let hidden = model.step(&Array::from_slice(&[1000], &[1, 1]), &mut cache)?;
                argmax_axis(&model.logits(&hidden)?, -1, None)?.eval()?;
            } else {
                round(model, &mut cache, &greedy, 1000, depth)?;
                cache.rollback(1)?;
            }
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        costs.push(best);
    }
    Ok(costs)
}

/// One verified round: the proposals the target kept, then the target's own next token.
struct Round {
    tokens: Vec<i32>,
    accepted: usize,
}

/// Propose `depth` tokens after `current`, feed them all, and keep what the target accepts.
/// The cache then holds `current` and every proposal; the caller rolls back what it drops.
fn round(model: &Model, cache: &mut Cache, sampler: &Sampler, current: i32, depth: usize) -> Result<Round> {
    let draft = model.draft.as_ref().context("no draft network")?;
    let current = Array::from_slice(&[current], &[1]);
    let proposal = draft.propose(model, cache.draft.as_mut().context("no draft cache")?, &current, depth, sampler.temp)?;
    // Let the GPU start on the proposal while the target's graph is built.
    transforms::async_eval([&proposal.tokens])?;
    let n = proposal.tokens.dim(0);
    let ids = concatenate(&[&current, &proposal.tokens], 0)?.reshape(&[1, n + 1])?;
    let hidden = model.step(&ids, cache)?;
    let logits = model.logits(&hidden)?; // [n + 1, V]: row i scores the token after ids[i]
    let ints = |a: &Array| -> Result<Vec<i32>> {
        let a = a.as_dtype(Dtype::Int32)?;
        a.eval()?;
        Ok(a.as_slice::<i32>().to_vec())
    };
    let Some(q) = &proposal.probs else {
        // Greedy: a proposal stands where it is the target's choice.
        let target = argmax_axis(&logits, -1, None)?;
        transforms::eval([&target, &proposal.tokens])?;
        let (target, drafts) = (ints(&target)?, ints(&proposal.tokens)?);
        let accepted = drafts.iter().zip(&target).take_while(|(d, t)| d == t).count();
        let mut tokens = drafts[..accepted].to_vec();
        tokens.push(target[accepted]);
        return Ok(Round { tokens, accepted });
    };
    // Sampled: accept a proposal x with probability min(1, p(x) / q(x)); at the first rejection
    // draw from the leftover max(p - q, 0) instead. The result is distributed as the target's.
    let p = sampler.probs(&logits)?;
    let (head, last) = (p.index((..n, ..)), p.index((n.., ..)));
    let at = proposal.tokens.reshape(&[n, 1])?;
    let ratio = minimum(&take_along_axis(&head, &at, -1)?.divide(&take_along_axis(q, &at, -1)?)?, Array::from_f32(1.0))?;
    let ok = random::uniform::<_, f32>(0.0, 1.0, &[n, 1], None)?.lt(&ratio)?;
    let leftover = maximum(&head.subtract(q)?, Array::from_f32(0.0))?;
    let residual = random::categorical(&leftover.log()?, -1, None, None)?;
    let bonus = random::categorical(&last.log()?, -1, None, None)?;
    transforms::eval([&ok, &residual, &bonus, &proposal.tokens])?;
    let drafts = ints(&proposal.tokens)?;
    let accepted = ints(&ok)?.iter().take_while(|&&v| v != 0).count();
    let mut tokens = drafts[..accepted].to_vec();
    tokens.push(if accepted < drafts.len() { ints(&residual)?[accepted] } else { ints(&bonus)?[0] });
    Ok(Round { tokens, accepted })
}

/// Prefill `prompt_ids`, then decode until an end token or `max_tokens`, handing each token to
/// `on_token` (false stops). Tokens the caller never saw are not left in the cache.
pub fn generate(
    model: &Model,
    cache: &mut Cache,
    sampler: &Sampler,
    eos: &[u32],
    prompt_ids: &[u32],
    max_tokens: usize,
    mut on_token: impl FnMut(u32) -> bool,
) -> Result<Stats> {
    let draft = model.draft.as_ref().context("no draft network")?;
    let ids: Vec<i32> = prompt_ids.iter().map(|&t| t as i32).collect();
    let feed = |token: &Array, cache: &mut Cache| -> Result<Array> {
        let hidden = model.step(&token.reshape(&[1, 1])?, cache)?;
        if let Some(dc) = &mut cache.draft {
            draft.flush_if_full(dc)?;
        }
        let next = sampler.sample(&model.logits(&hidden)?)?;
        transforms::async_eval([&next])?;
        Ok(next)
    };

    let t0 = Instant::now();
    let hidden = model.prefill(&Array::from_slice(&ids, &[1, ids.len() as i32]), cache)?;
    let seed = sampler.sample(&model.logits(&hidden.index((.., hidden.dim(1) - 1.., ..)))?)?;
    transforms::async_eval([&seed])?;

    // Either a token is in flight (sampled; not read, handed out or fed yet), or `current` is
    // the last token handed out, not fed yet.
    let mut inflight = Some(seed);
    let mut current = 0i32;
    let mut control = Controller::new(draft.limit());
    let mut n = 0;
    let mut t_gen = Instant::now();
    let mut prompt_secs = 0.0;
    let mut ended = false;
    // Hand one token out; false ends the turn.
    let mut emit = |token: i32, n: &mut usize, ended: &mut bool| -> bool {
        if *n == 0 {
            prompt_secs = t0.elapsed().as_secs_f64();
            t_gen = Instant::now();
        }
        *n += 1;
        if eos.contains(&(token as u32)) {
            *ended = true;
            return false;
        }
        on_token(token as u32) && *n < max_tokens
    };
    let mut learned = std::mem::take(&mut cache.draft.as_mut().context("no draft cache")?.depth);
    while n < max_tokens {
        let depth = control.depth.min(max_tokens - n - 1);
        if depth == 0 {
            // Plain decoding, pipelined: the next step is queued before this token is read.
            control.begin_round(&mut learned);
            let token = match inflight.take() {
                Some(t) => t,
                None => feed(&Array::from_slice(&[current], &[1]), cache)?,
            };
            let next = if n + 1 < max_tokens { Some(feed(&token, cache)?) } else { None };
            current = token.item::<u32>() as i32;
            let go_on = emit(current, &mut n, &mut ended);
            inflight = next;
            control.end_round(&mut learned, 0, 0, 0);
            if !go_on {
                // The token is in the cache if its step was queued; that step's sample is dropped.
                if inflight.take().is_some() {
                    current = -1;
                }
                break;
            }
            continue;
        }
        if let Some(token) = inflight.take() {
            // Leave plain decoding: the token in flight becomes the current one.
            current = token.item::<u32>() as i32;
            control.untimed_round();
            if !emit(current, &mut n, &mut ended) {
                break;
            }
            continue;
        }
        control.begin_round(&mut learned);
        let Round { tokens, accepted } = round(model, cache, sampler, current, depth)?;
        // Hand the tokens out, then keep only what was taken: `current` and the proposals
        // before the last token handed out, which becomes the new current token.
        let mut last = 0;
        let mut go_on = true;
        for (i, &token) in tokens.iter().enumerate() {
            last = i;
            go_on = emit(token, &mut n, &mut ended);
            if !go_on {
                break;
            }
        }
        cache.rollback(last as i32 + 1)?;
        current = tokens[last];
        // An accepted end token says nothing about the positions after it.
        let observed = if ended && last < accepted { last + 1 } else { depth };
        control.end_round(&mut learned, depth, accepted.min(observed), observed);
        if !go_on {
            break;
        }
    }
    // A token handed out but not fed opens the next forward.
    if current >= 0 && inflight.is_none() && n > 0 {
        cache.pending = Some(current);
    }
    if std::env::var_os("INKLING_STATS").is_some() {
        control.report(&learned, n);
    }
    cache.draft.as_mut().context("no draft cache")?.depth = learned;
    let gen_secs = t_gen.elapsed().as_secs_f64();
    Ok(Stats {
        prompt_tokens: prompt_ids.len(),
        prompt_tps: prompt_ids.len() as f64 / prompt_secs.max(1e-9),
        gen_tokens: n,
        gen_tps: n.saturating_sub(1) as f64 / gen_secs.max(1e-9),
        eos_in_cache: ended,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_is_linear_between_samples_and_flat_beyond() {
        let mut cost = CostModel::default();
        assert!(!cost.ready());
        cost.observe(0, 20.0);
        cost.observe(4, 40.0);
        assert!(cost.ready());
        assert_eq!(cost.cost(2), 30.0);
        assert_eq!(cost.cost(7), 40.0);
        // One stalled round moves the average by at most a quarter of itself, times the weight.
        cost.observe(0, 500.0);
        assert!((cost.cost(0) - 21.5).abs() < 1e-9);
    }

    #[test]
    fn acceptance_counts_only_positions_that_were_reached() {
        let mut acc = AcceptanceModel::default();
        // Three proposed, the first accepted: position 3 was never reached.
        acc.observe(3, 1);
        assert_eq!((acc.acceptance(1), acc.acceptance(2)), (1.0, 0.0));
        assert_eq!(acc.frontier(), 2);
        assert_eq!(acc.expected(3), 2.0);
        // Positions past the first share one rate.
        acc.observe(3, 3);
        assert!((acc.acceptance(3) - acc.acceptance(2)).abs() < 1e-12);
    }

    #[test]
    fn the_controller_settles_on_the_depth_with_the_best_rate() {
        // 20 ms plain, 5 ms per proposal; the first proposal is accepted 9 times in 10, later
        // ones never: depth 1 gives 1.9 tokens per 25 ms, better than plain or deeper.
        let mut learned = Learned::default();
        for depth in 0..=7 {
            learned.cost.observe(depth, 20.0 + 5.0 * depth as f64);
        }
        let mut control = Controller { fixed: None, ..Controller::new(7) };
        assert_eq!(control.depth, 7, "a request starts with a whole block");
        let mut counts = [0usize; 8];
        for round in 0..200 {
            let depth = control.depth;
            counts[depth] += 1;
            let accepted = if depth == 0 { 0 } else { (round % 10 != 5) as usize };
            control.end_round(&mut learned, depth, accepted, depth);
        }
        assert!(counts[1] > 150, "rounds by depth: {counts:?}");
    }
}
