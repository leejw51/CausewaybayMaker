//! A Python helper kept running in the background (python/see.py, python/speak.py): the model
//! loads once, then one JSON object per line goes in and one comes back.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use anyhow::{Context, Result, bail};
use serde_json::Value;

pub struct Helper {
    child: Child,
    stdin: ChildStdin,
    out: BufReader<ChildStdout>,
    name: String,
    ready: bool,
    /// Model facts from the ready line: "Kokoro-82M-bf16 · mlx-audio 0.5.9 · ~/.cache/... · 330 MB".
    pub info: String,
    /// Just name and version: "Kokoro-82M-bf16 · mlx-audio 0.5.9".
    pub info_short: String,
}

impl Helper {
    pub fn start(python: &Path, script: &Path, args: &[String], verbose: bool) -> Result<Self> {
        let name = script.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let mut child = Command::new(python)
            .arg("-I")
            .arg(script)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(if verbose { Stdio::inherit() } else { Stdio::null() })
            .process_group(0)
            .spawn()
            .with_context(|| format!("start {} {}", python.display(), script.display()))?;
        let stdin = child.stdin.take().unwrap();
        let out = BufReader::new(child.stdout.take().unwrap());
        Ok(Self { child, stdin, out, name, ready: false, info: String::new(), info_short: String::new() })
    }

    fn read(&mut self) -> Result<Value> {
        let mut line = String::new();
        if self.out.read_line(&mut line)? == 0 {
            bail!("{} exited (run with --verbose to see why)", self.name);
        }
        serde_json::from_str(&line).with_context(|| format!("{} said {line:?}", self.name))
    }

    /// Block until the helper reports that its model is loaded; returns the load time.
    pub fn wait_ready(&mut self) -> Result<f64> {
        if self.ready {
            return Ok(0.0);
        }
        let v = self.read()?;
        if v["ready"].as_bool() != Some(true) {
            bail!("{} did not start: {v}", self.name);
        }
        self.ready = true;
        self.info = describe_ready(&v);
        self.info_short = describe_ready_short(&v);
        Ok(v["seconds"].as_f64().unwrap_or(0.0))
    }

    /// Send one request and return the reply object (an `error` field becomes an Err).
    pub fn call(&mut self, request: &Value) -> Result<Value> {
        self.call_with_progress(request, |_| {})
    }

    /// Like `call`, but intermediate `{"progress": n, "total": m}` lines go to `on_progress`.
    pub fn call_with_progress(&mut self, request: &Value, mut on_progress: impl FnMut(&Value)) -> Result<Value> {
        self.wait_ready()?;
        writeln!(self.stdin, "{request}")?;
        self.stdin.flush()?;
        loop {
            let v = self.read()?;
            if v.get("progress").is_some() {
                on_progress(&v);
                continue;
            }
            if let Some(e) = v["error"].as_str() {
                bail!("{}: {e}", self.name);
            }
            return Ok(v);
        }
    }
}

/// Name and version from a helper's ready message: "org/m · lib 1.2".
fn describe_ready_short(v: &Value) -> String {
    let mut parts = vec![v["model"].as_str().unwrap_or("?").to_string()];
    if let Some(ver) = v["version"].as_str() {
        parts.push(ver.to_string());
    }
    parts.join(" · ")
}

/// One line of model facts from a helper's ready message.
fn describe_ready(v: &Value) -> String {
    let mut parts = vec![v["model"].as_str().unwrap_or("?").to_string()];
    if let Some(ver) = v["version"].as_str() {
        parts.push(ver.to_string());
    }
    match v["path"].as_str() {
        Some(p) => parts.push(qwen35::hf::tilde(std::path::Path::new(p))),
        None => parts.push("not downloaded yet".into()),
    }
    if let Some(b) = v["size_bytes"].as_u64().filter(|&b| b > 0) {
        parts.push(qwen35::hf::human_size(b));
    }
    parts.join(" · ")
}

impl Drop for Helper {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::Helper;
    use serde_json::json;

    /// A stand-in for see.py / speak.py: says ready, then echoes requests or errors on demand.
    fn fake_helper() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("jarvis-helper-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("fake.py");
        std::fs::write(
            &script,
            "import json, sys\nprint(json.dumps({'ready': True, 'seconds': 0.5}), flush=True)\n\
             for line in sys.stdin:\n    r = json.loads(line)\n    \
             print(json.dumps({'progress': 1, 'total': 2}), flush=True)\n    \
             print(json.dumps({'error': 'boom'} if r.get('fail') else {'echo': r}), flush=True)\n",
        )
        .unwrap();
        script
    }

    #[test]
    fn protocol_round_trip() {
        let script = fake_helper();
        let mut h = Helper::start(std::path::Path::new("python3"), &script, &["--x".into()], false).unwrap();
        assert_eq!(h.wait_ready().unwrap(), 0.5);
        let v = h.call(&json!({ "text": "hi" })).unwrap();
        assert_eq!(v["echo"]["text"], "hi");
        let err = h.call(&json!({ "fail": true })).unwrap_err().to_string();
        assert!(err.contains("boom"), "{err}");
        // Still serving after an error.
        assert_eq!(h.call(&json!({ "n": 2 })).unwrap()["echo"]["n"], 2);
        // Progress lines reach the callback and never leak into the reply.
        let mut seen = Vec::new();
        let v = h.call_with_progress(&json!({ "n": 3 }), |p| seen.push((p["progress"].as_u64(), p["total"].as_u64()))).unwrap();
        assert_eq!(v["echo"]["n"], 3);
        assert_eq!(seen, vec![(Some(1), Some(2))]);
    }

    #[test]
    fn ready_facts_are_summarised() {
        let v = json!({"ready": true, "model": "org/m", "version": "lib 1.2", "path": "/opt/m", "size_bytes": 2_000_000_000u64});
        assert_eq!(super::describe_ready(&v), "org/m · lib 1.2 · /opt/m · 2.0 GB");
        assert_eq!(super::describe_ready_short(&v), "org/m · lib 1.2");
        let v = json!({"ready": true, "model": "org/m", "path": null, "size_bytes": 0});
        assert_eq!(super::describe_ready(&v), "org/m · not downloaded yet");
    }

    #[test]
    fn missing_script_fails_to_start_or_report() {
        let bad = std::path::Path::new("/nonexistent/helper.py");
        match Helper::start(std::path::Path::new("python3"), bad, &[], false) {
            Ok(mut h) => assert!(h.wait_ready().is_err()),
            Err(_) => {}
        }
    }
}
