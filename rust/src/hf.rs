//! Model files: a local directory, the Hugging Face cache, or a download with the `hf` CLI.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

pub fn hf_cache_dir() -> PathBuf {
    if let Ok(p) = std::env::var("HF_HUB_CACHE") {
        return p.into();
    }
    let home = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/huggingface"));
    home.join("hub")
}

pub fn cached_snapshot(repo: &str, marker: &str) -> Option<PathBuf> {
    let base = hf_cache_dir().join(format!("models--{}", repo.replace('/', "--")));
    let rev = std::fs::read_to_string(base.join("refs/main")).ok()?;
    let dir = base.join("snapshots").join(rev.trim());
    dir.join(marker).exists().then_some(dir)
}

/// Local dir, else the Hugging Face cache, else download `files` of `repo` with the `hf` CLI.
pub fn resolve(repo: &str, marker: &str, files: &[&str]) -> Result<PathBuf> {
    let p = Path::new(repo);
    if p.join(marker).exists() {
        return Ok(p.to_path_buf());
    }
    if let Some(dir) = cached_snapshot(repo, marker) {
        return Ok(dir);
    }
    eprintln!("Downloading {repo} from Hugging Face...");
    let status = Command::new("hf")
        .args(["download", repo])
        .args(files)
        .status()
        .context("`hf` CLI not found; run `make setup` or `pip install huggingface_hub[cli]`")?;
    if !status.success() {
        bail!("hf download {repo} failed");
    }
    cached_snapshot(repo, marker).context("model not found in cache after download")
}


/// Total size of the files under `dir` (symlinks into the Hugging Face blob store followed).
pub fn dir_size(dir: &Path) -> u64 {
    fn walk(dir: &Path, total: &mut u64) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                let p = e.path();
                match std::fs::metadata(&p) {
                    Ok(m) if m.is_dir() => walk(&p, total),
                    Ok(m) => *total += m.len(),
                    Err(_) => {}
                }
            }
        }
    }
    let mut total = 0;
    walk(dir, &mut total);
    total
}

/// "1.6 GB" / "640 MB".
pub fn human_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1e9)
    } else {
        format!("{} MB", bytes / 1_000_000)
    }
}

/// `$HOME/...` shown as `~/...`.
pub fn tilde(path: &Path) -> String {
    let p = path.to_string_lossy();
    match std::env::var("HOME") {
        Ok(home) if !home.is_empty() && p.starts_with(&home) => format!("~{}", &p[home.len()..]),
        _ => p.into_owned(),
    }
}

/// The snapshot's commit, shortened: " · rev 8b2b98c" (empty for a plain directory).
fn revision(dir: &Path) -> String {
    dir.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .filter(|n| n.len() >= 7 && n.chars().all(|c| c.is_ascii_hexdigit()))
        .map(|n| format!(" · rev {}", &n[..7]))
        .unwrap_or_default()
}

/// Name and version of a local model: "mlx-community/whisper-large-v3-turbo · rev 8b2b98c".
pub fn describe_short(repo: &str, dir: &Path) -> String {
    format!("{repo}{}", revision(dir))
}

/// Everything about a local model: repo, revision, where it is, how big.
/// "mlx-community/whisper-large-v3-turbo · rev 8b2b98c · ~/.cache/huggingface/hub/... · 1.6 GB"
pub fn describe(repo: &str, dir: &Path) -> String {
    format!("{} · {} · {}", describe_short(repo, dir), tilde(dir), human_size(dir_size(dir)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_paths_and_descriptions() {
        assert_eq!(human_size(1_600_000_000), "1.6 GB");
        assert_eq!(human_size(640_000_000), "640 MB");
        let home = std::env::var("HOME").unwrap();
        assert_eq!(tilde(Path::new(&format!("{home}/x/y"))), "~/x/y");
        assert_eq!(tilde(Path::new("/opt/z")), "/opt/z");
        let dir = std::env::temp_dir().join(format!("hf-describe-{}", std::process::id())).join("snapshots").join("8b2b98c00a6b4d29");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("w.bin"), vec![0u8; 2_500_000]).unwrap();
        assert_eq!(describe_short("org/model", &dir), "org/model · rev 8b2b98c");
        let d = describe("org/model", &dir);
        assert!(d.starts_with("org/model · rev 8b2b98c · "), "{d}");
        assert!(d.ends_with(" · 2 MB"), "{d}");
        assert_eq!(dir_size(&dir), 2_500_000);
        let _ = std::fs::remove_dir_all(dir.parent().unwrap().parent().unwrap());
    }

    #[test]
    fn resolve_prefers_a_local_directory() {
        let dir = std::env::temp_dir().join(format!("hf-test-local-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        assert_eq!(resolve(dir.to_str().unwrap(), "config.json", &[]).unwrap(), dir);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cached_snapshot_follows_refs_main() {
        let cache = std::env::temp_dir().join(format!("hf-test-cache-{}", std::process::id()));
        let repo = cache.join("models--org--model");
        let snap = repo.join("snapshots/abc123");
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::write(repo.join("refs/main"), "abc123\n").unwrap();
        std::fs::write(snap.join("config.json"), "{}").unwrap();
        // SAFETY: tests in this module run single-threaded with respect to this variable.
        unsafe { std::env::set_var("HF_HUB_CACHE", &cache) };
        assert_eq!(cached_snapshot("org/model", "config.json"), Some(snap.clone()));
        assert_eq!(cached_snapshot("org/model", "tokenizer.json"), None); // marker missing
        assert_eq!(cached_snapshot("org/other", "config.json"), None);
        unsafe { std::env::remove_var("HF_HUB_CACHE") };
        let _ = std::fs::remove_dir_all(&cache);
    }
}
