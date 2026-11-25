//! Check the Inkling 2 port against Ollama: print the top next-token log-probabilities for a raw
//! prompt, then greedy-decode a few tokens through the cache.
//!
//!   cargo run --release --example inkling -- [--model inkling-2-mini:26b-a5b-nvfp4] [--tokens 12] "<raw prompt>"
//!   ... --score continuation.txt "<raw prompt>"   teacher-force a continuation (e.g. Ollama's) and
//!                                                 report the log-probability of each of its tokens
//!   ... --chunked 700 "<raw prompt>"              compare one-shot prefill with a 700-token prefill
//!                                                 followed by one token at a time
//!   ... --split 900 "<raw prompt>"                compare two forwards (900 tokens, then the rest
//!                                                 at once) with 900 tokens then one at a time
//!   INKLING_PROFILE=1 ...                         time a decode step (graph build and evaluation);
//!                                                 INKLING_ROWS=4 feeds 4 tokens per step
//!   INKLING_MICRO=1 ...                           latency of each kernel of a decode step
//!   INKLING_ROUND=1 ...                           time the parts of a speculative round by depth
//!   INKLING_SHARED=1 ...                          check the shared-expert products against MLX's
//!   INKLING_ROLLBACK=1 ...                        check that taking rows back out of the cache
//!                                                 matches never having fed them (long prompt)
//!   INKLING_PREFILL=325,702,1860 ...              prompt reading time at these lengths (best of
//!                                                 three; INKLING_PREFILL_ALL=1 prints each);
//!                                                 with INKLING_TRACE=1, the time by stage
//!                                                 (every stage is evaluated on its own, which
//!                                                 adds about a quarter to the total)
//!   INKLING_VISION=photo.jpg ...                  image encoding time, six times over
//!                                                 (INKLING_VISION_SAVE=file writes the embeddings)

use std::time::Instant;

use anyhow::Result;
use mlx_rs::Array;
use mlx_rs::ops::indexing::argmax_axis;
use qwen35::inkling::{Model, ollama::Manifest};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |flag: &str, default: &str| -> String {
        match args.iter().position(|a| a == flag) {
            Some(i) => {
                let v = args.remove(i + 1);
                args.remove(i);
                v
            }
            None => default.to_string(),
        }
    };
    let model_name = take("--model", "inkling-2-mini:26b-a5b-nvfp4");
    let n: usize = take("--tokens", "12").parse()?;
    let score = take("--score", "");
    let chunked: usize = take("--chunked", "0").parse()?;
    let split: usize = take("--split", "0").parse()?;
    let prompt = args.first().cloned().unwrap_or_else(|| {
        "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>What is the capital of France?<|end_message|>".into()
    });

    let t0 = Instant::now();
    let manifest = Manifest::load(&model_name)?;
    let tok = Tokenizer::from_file(manifest.file("tokenizer.json")?).map_err(anyhow::Error::msg)?;
    let model = Model::load(&manifest)?;
    eprintln!("loaded in {:.1}s, {:.1} GB active", t0.elapsed().as_secs_f64(), mlx_rs::memory::active_memory().unwrap_or(0) as f64 / 1e9);

    let ids: Vec<i32> = tok.encode(prompt.as_str(), false).map_err(anyhow::Error::msg)?.get_ids().iter().map(|&i| i as i32).collect();
    eprintln!("prompt: {} tokens", ids.len());
    let logprobs = |logits: &Array| -> Result<Vec<f32>> {
        let lp = logits.subtract(mlx_rs::ops::logsumexp_axis(logits, -1, true)?)?;
        lp.eval()?;
        Ok(lp.as_slice::<f32>().to_vec())
    };
    let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
    if std::env::var_os("INKLING_SHARED").is_some() {
        println!("shared-expert products against MLX's: largest relative difference {:.2e}", model.check_shared_products()?);
        return Ok(());
    }
    if std::env::var_os("INKLING_MICRO").is_some() {
        return model.microbench();
    }
    if let Ok(path) = std::env::var("INKLING_VISION") {
        // Image encoding time, after the first (which loads the encoder).
        let vision = model.vision()?;
        let pixels = vision.read(std::path::Path::new(&path))?;
        for i in 0..6 {
            let t = Instant::now();
            let out = vision.encode(&pixels)?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let f = out.as_dtype(mlx_rs::Dtype::Float32)?;
            if let Ok(save) = std::env::var("INKLING_VISION_SAVE") {
                f.eval()?;
                let bytes: Vec<u8> = f.as_slice::<f32>().iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(&save, bytes)?;
            }
            println!("encode {i}: {} tokens in {ms:.1} ms (mean |x| {:.6}, sum {:.4})", out.dim(0), f.abs()?.mean(None)?.item::<f32>(), f.sum(None)?.item::<f32>());
        }
        return Ok(());
    }
    if std::env::var_os("INKLING_PREFILL").is_some() {
        // Prompt reading speed at several lengths (the prompt's tokens, repeated).
        let lengths: Vec<usize> = std::env::var("INKLING_PREFILL").unwrap().split(',').filter_map(|v| v.parse().ok()).collect();
        for &len in &lengths {
            let fed: Vec<i32> = ids.iter().cycle().take(len).copied().collect();
            let mut best = f64::MAX;
            for _ in 0..3 {
                let mut cache = model.make_cache();
                let t = Instant::now();
                let logits = model.forward(&Array::from_slice(&fed, &[1, len as i32]), &mut cache)?;
                mlx_rs::transforms::eval([&logits])?;
                cache.eval_state()?;
                best = best.min(t.elapsed().as_secs_f64());
                if std::env::var_os("INKLING_PREFILL_ALL").is_some() {
                    println!("  {len} tokens: {:.3} s", t.elapsed().as_secs_f64());
                }
            }
            println!("prefill {len} tokens: {:.3} s, {:.0} tok/s; peak memory so far {:.1} GB", best, len as f64 / best, mlx_rs::memory::peak_memory().unwrap_or(0) as f64 / 1e9);
        }
        return Ok(());
    }
    if std::env::var_os("INKLING_PROFILE").is_some() {
        let mut cache = model.make_cache();
        let mut logits = model.forward(&Array::from_slice(&ids, &[1, ids.len() as i32]), &mut cache)?;
        logits.eval()?;
        let (mut build, mut run) = (0.0, 0.0);
        let steps = 40;
        let rows: usize = std::env::var("INKLING_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        for i in 0..steps {
            let next = argmax_axis(&logits, -1, None)?.item::<u32>();
            let t = Instant::now();
            logits = model.forward(&Array::from_slice(&vec![next as i32; rows], &[1, rows as i32]), &mut cache)?;
            let b = t.elapsed().as_secs_f64();
            logits.eval()?;
            if i >= 5 {
                build += b;
                run += t.elapsed().as_secs_f64() - b;
            }
        }
        let n = (steps - 5) as f64;
        println!("per step of {rows} token(s): graph build {:.1} ms, eval {:.1} ms, total {:.1} ms ({:.1} tok/s unpipelined)", build / n * 1e3, run / n * 1e3, (build + run) / n * 1e3, n / (build + run));
        return Ok(());
    }
    if let Some(draft) = model.draft.as_ref().filter(|_| std::env::var_os("INKLING_ROUND").is_some()) {
        // Time the parts of a speculative round at each depth: the proposal, then the pass
        // that verifies it (rows and vocabulary scores).
        let mut cache = model.make_cache();
        model.forward(&Array::from_slice(&ids, &[1, ids.len() as i32]), &mut cache)?.eval()?;
        let temp: f32 = std::env::var("INKLING_TEMP").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
        for depth in [1usize, 2, 3, 4, 7] {
            let (mut propose, mut verify, mut score, mut undo) = (0.0, 0.0, 0.0, 0.0);
            let reps = 12;
            for i in 0..reps {
                let current = Array::from_slice(&[ids[i % ids.len()]], &[1]);
                let t = Instant::now();
                let p = draft.propose(&model, cache.draft.as_mut().unwrap(), &current, depth, temp)?;
                p.tokens.eval()?;
                let a = t.elapsed().as_secs_f64();
                let row = mlx_rs::ops::concatenate(&[&current, &p.tokens], 0)?.reshape(&[1, depth as i32 + 1])?;
                let hidden = model.step(&row, &mut cache)?;
                hidden.eval()?;
                let b = t.elapsed().as_secs_f64();
                argmax_axis(&model.logits(&hidden)?, -1, None)?.eval()?;
                let c = t.elapsed().as_secs_f64();
                cache.rollback(1)?;
                let t2 = Instant::now();
                cache.eval_state()?;
                if i >= 2 {
                    undo += t2.elapsed().as_secs_f64();
                    propose += a;
                    verify += b - a;
                    score += c - b;
                }
            }
            let n = (reps - 2) as f64;
            println!(
                "depth {depth}: propose {:.1} ms, verify {} rows {:.1} ms, vocabulary {:.1} ms, taking rows back {:.1} ms, total {:.1} ms",
                propose / n * 1e3,
                depth + 1,
                verify / n * 1e3,
                score / n * 1e3,
                undo / n * 1e3,
                (propose + verify + score + undo) / n * 1e3
            );
        }
        return Ok(());
    }
    if std::env::var_os("INKLING_ROLLBACK").is_some() {
        // Feed 8 rows and take back all but j. The cache must then equal, bit for bit, one that
        // was fed the same j rows followed by different rows and rolled back the same way: the
        // dropped rows leave no trace. (A pass of only j rows differs by rounding, because rows
        // are multiplied in different batches; that difference is printed for scale.) The bases
        // put the 8 rows across a KV buffer growth (508), a local-window trim (1276) and neither.
        let mut failed = false;
        for (base, first) in [(508usize, 250usize), (1276, 1276), (ids.len() - 9, ids.len() - 9)] {
            if base + 9 > ids.len() || base < 16 {
                println!("base {base}: prompt too short, skipped");
                continue;
            }
            let rows = &ids[base..base + 8];
            let probe = ids[base + 8];
            let filled = || -> Result<qwen35::inkling::Cache> {
                let mut cache = model.make_cache();
                model.forward(&Array::from_slice(&ids[..first], &[1, first as i32]), &mut cache)?.eval()?;
                if base > first {
                    model.forward(&Array::from_slice(&ids[first..base], &[1, (base - first) as i32]), &mut cache)?.eval()?;
                }
                Ok(cache)
            };
            for j in [1usize, 2, 3, 5, 8] {
                let mut a = filled()?;
                model.step(&Array::from_slice(rows, &[1, 8]), &mut a)?.eval()?;
                a.rollback(j as i32)?;
                // Same first j rows, then other tokens.
                let mut other = rows.to_vec();
                other[j..].copy_from_slice(&ids[base - 8 + j..base]);
                let mut b = filled()?;
                model.step(&Array::from_slice(&other, &[1, 8]), &mut b)?.eval()?;
                b.rollback(j as i32)?;
                let state = a.debug_diff(&b)?;
                let mut c = filled()?;
                model.step(&Array::from_slice(&rows[..j], &[1, j as i32]), &mut c)?.eval()?;
                let next = |cache: &mut qwen35::inkling::Cache| logprobs(&model.forward(&Array::from_slice(&[probe], &[1, 1]), cache)?);
                let (la, lb, lc) = (next(&mut a)?, next(&mut b)?, next(&mut c)?);
                let diff = |x: &[f32], y: &[f32]| x.iter().zip(y).filter(|(a, _)| **a > -8.0).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                let exact = state.is_empty() && diff(&la, &lb) == 0.0;
                failed |= !exact;
                println!(
                    "base {base:4}, keep {j} of 8: {}; against a pass of {j} rows the next logprobs differ by {:.3}{}",
                    if exact { "exact".to_string() } else { format!("MISMATCH (logprobs {:.4}) {state}", diff(&la, &lb)) },
                    diff(&la, &lc),
                    if argmax(&la) == argmax(&lc) { "" } else { ", other top token" },
                );
            }
        }
        if failed {
            anyhow::bail!("rollback left traces of dropped rows");
        }
        return Ok(());
    }
    if split > 0 {
        let run = |stepwise: bool| -> Result<Vec<f32>> {
            let mut cache = model.make_cache();
            model.forward(&Array::from_slice(&ids[..split], &[1, split as i32]), &mut cache)?.eval()?;
            let rest = &ids[split..];
            let mut logits = if stepwise { None } else { Some(model.forward(&Array::from_slice(rest, &[1, rest.len() as i32]), &mut cache)?) };
            if stepwise {
                for &id in rest {
                    let l = model.forward(&Array::from_slice(&[id], &[1, 1]), &mut cache)?;
                    l.eval()?;
                    logits = Some(l);
                }
            }
            logprobs(&logits.unwrap())
        };
        let (two, step) = (run(false)?, run(true)?);
        let diff = two.iter().zip(&step).filter(|(a, _)| **a > -8.0).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let show = |id: u32, v: &[f32]| format!("{:?} {:.3}", tok.decode(&[id], false).unwrap_or_default(), v[id as usize]);
        println!("{split} + {} at once : {}", ids.len() - split, show(argmax(&two), &two));
        println!("{split} + stepwise     : {}", show(argmax(&step), &step));
        println!("max |logprob difference| over tokens above -8: {diff:.4}");
        return Ok(());
    }
    if chunked > 0 {
        let whole = logprobs(&model.forward(&Array::from_slice(&ids, &[1, ids.len() as i32]), &mut model.make_cache())?)?;
        let mut cache = model.make_cache();
        let mut logits = model.forward(&Array::from_slice(&ids[..chunked], &[1, chunked as i32]), &mut cache)?;
        for &id in &ids[chunked..] {
            logits = model.forward(&Array::from_slice(&[id], &[1, 1]), &mut cache)?;
            logits.eval()?; // as decoding does; keeps the graph from spanning hundreds of steps
        }
        let step = logprobs(&logits)?;
        let diff = whole.iter().zip(&step).filter(|(a, _)| **a > -8.0).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let show = |id: u32, v: &[f32]| format!("{:?} {:.3}", tok.decode(&[id], false).unwrap_or_default(), v[id as usize]);
        println!("one-shot  : {}", show(argmax(&whole), &whole));
        println!("stepwise  : {}", show(argmax(&step), &step));
        println!("max |logprob difference| over tokens above -8: {diff:.4}");
        return Ok(());
    }
    if !score.is_empty() {
        let text = std::fs::read_to_string(&score)?;
        let cont: Vec<u32> = tok.encode(text.as_str(), false).map_err(anyhow::Error::msg)?.get_ids().to_vec();
        let mut cache = model.make_cache();
        let first = model.forward(&Array::from_slice(&ids, &[1, ids.len() as i32]), &mut cache)?;
        // INKLING_ROWS=n scores n tokens per pass, as a speculative round does.
        let rows: usize = std::env::var("INKLING_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let mut pending = vec![logprobs(&first)?];
        let (mut agree, mut total_lp) = (0, 0.0f32);
        for (i, &id) in cont.iter().enumerate() {
            if pending.is_empty() {
                // Feed the tokens scored so far that the cache has not seen, up to `rows`.
                let fed = &cont[i - 1..(i - 1 + rows).min(cont.len())];
                let fed: Vec<i32> = fed.iter().map(|&t| t as i32).collect();
                let hidden = model.step(&Array::from_slice(&fed, &[1, fed.len() as i32]), &mut cache)?;
                let all = model.logits(&hidden)?;
                for r in 0..fed.len() as i32 {
                    pending.push(logprobs(&mlx_rs::ops::indexing::IndexOp::index(&all, (r..r + 1, ..)))?);
                }
                pending.reverse();
            }
            let lp = pending.pop().unwrap();
            let best = argmax(&lp);
            agree += (best == id) as usize;
            total_lp += lp[id as usize];
            println!(
                "{i:3} {:<20} {:8.3}{}",
                format!("{:?}", tok.decode(&[id], false).unwrap_or_default()),
                lp[id as usize],
                if best == id { String::new() } else { format!("   (ours: {:?} {:.3})", tok.decode(&[best], false).unwrap_or_default(), lp[best as usize]) }
            );
        }
        println!("argmax agrees on {agree}/{} tokens; mean logprob of the given tokens {:.4}", cont.len(), total_lp / cont.len() as f32);
        return Ok(());
    }
    let mut cache = model.make_cache();
    let t1 = Instant::now();
    let mut logits = model.forward(&Array::from_slice(&ids, &[1, ids.len() as i32]), &mut cache)?;
    let mut out = Vec::new();
    for step in 0..n {
        let logp = logits.subtract(mlx_rs::ops::logsumexp_axis(&logits, -1, true)?)?;
        let order = mlx_rs::ops::argsort_axis(logp.negative()?, -1)?;
        order.eval()?;
        let order: Vec<u32> = order.as_slice::<u32>()[..5].to_vec();
        let lp = logp.as_slice::<f32>().to_vec();
        let next = argmax_axis(&logits, -1, None)?.item::<u32>();
        let show = |id: u32| format!("{:?} {:.3}", tok.decode(&[id], false).unwrap_or_default(), lp[id as usize]);
        println!("{step:2} {:<28} top: {}", show(next), order.iter().map(|&i| show(i)).collect::<Vec<_>>().join(" | "));
        out.push(next);
        if step == 0 {
            eprintln!("prefill {:.2}s", t1.elapsed().as_secs_f64());
        }
        if step + 1 < n {
            logits = model.forward(&Array::from_slice(&[next as i32], &[1, 1]), &mut cache)?;
        }
    }
    let secs = t1.elapsed().as_secs_f64();
    println!("{:?}", tok.decode(&out, false).unwrap_or_default());
    eprintln!("{} tokens in {:.2}s; peak {:.1} GB", n, secs, mlx_rs::memory::peak_memory().unwrap_or(0) as f64 / 1e9);
    Ok(())
}
