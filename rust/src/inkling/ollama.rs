//! Read a model straight out of Ollama's store: the manifest maps each tensor name and config
//! file to a blob, and every tensor blob is a small safetensors file.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::Value;

#[derive(Clone)]
pub struct Manifest {
    pub name: String,
    pub blobs: PathBuf,
    /// Tensor and config-file name -> blob path.
    pub files: HashMap<String, PathBuf>,
    /// Total size of all layers, bytes.
    pub size: u64,
}

fn models_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("OLLAMA_MODELS") {
        return PathBuf::from(dir);
    }
    PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".ollama/models")
}

/// Is `spec` an Ollama model name ("inkling-2-mini:26b-a5b-nvfp4", optionally "ollama:...")?
pub fn manifest_path(spec: &str) -> Option<PathBuf> {
    let spec = spec.strip_prefix("ollama:").unwrap_or(spec);
    if spec.contains('/') && !spec.starts_with("registry.") || spec.is_empty() {
        return None;
    }
    let (name, tag) = spec.split_once(':').unwrap_or((spec, "latest"));
    let p = models_dir().join("manifests/registry.ollama.ai/library").join(name).join(tag);
    p.is_file().then_some(p)
}

/// MLX keeps every tensor file open until its array is evaluated, and this model is hundreds of
/// files, while a terminal's default limit is 256 open files: raise the limit to fit them.
fn allow_open_files(count: usize) {
    let mut limit = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
    // SAFETY: getrlimit and setrlimit only read and write the struct passed to them.
    unsafe {
        if libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) != 0 {
            return;
        }
        let wanted = (count as libc::rlim_t + 256).min(limit.rlim_max);
        if limit.rlim_cur < wanted {
            limit.rlim_cur = wanted;
            libc::setrlimit(libc::RLIMIT_NOFILE, &limit);
        }
    }
}

impl Manifest {
    pub fn load(spec: &str) -> Result<Self> {
        let path = manifest_path(spec).with_context(|| format!("no Ollama model {spec} (try `ollama pull {spec}`)"))?;
        let root: Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
        let blobs = models_dir().join("blobs");
        let mut files = HashMap::new();
        let mut size = 0;
        for layer in root["layers"].as_array().context("manifest has no layers")? {
            size += layer["size"].as_u64().unwrap_or(0);
            if let (Some(name), Some(digest)) = (layer["name"].as_str(), layer["digest"].as_str()) {
                files.insert(name.to_string(), blobs.join(digest.replace(':', "-")));
            }
        }
        if files.is_empty() {
            bail!("{spec} is not a safetensors (MLX) model in Ollama's store");
        }
        allow_open_files(files.len());
        Ok(Self { name: spec.strip_prefix("ollama:").unwrap_or(spec).to_string(), blobs, files, size })
    }

    pub fn file(&self, name: &str) -> Result<&Path> {
        self.files.get(name).map(|p| p.as_path()).with_context(|| format!("{} has no {name}", self.name))
    }

    /// A path to a tensor blob that MLX will open: it picks the loader by file extension, so
    /// link the blob under a `.safetensors` name in a temporary folder.
    pub fn safetensors(&self, blob: &Path) -> Result<PathBuf> {
        let dir = std::env::temp_dir().join("ollama-safetensors");
        std::fs::create_dir_all(&dir)?;
        let name = blob.file_name().context("blob path has no file name")?;
        let link = dir.join(name).with_extension("safetensors");
        if std::fs::read_link(&link).ok().as_deref() != Some(blob) {
            let _ = std::fs::remove_file(&link);
            std::os::unix::fs::symlink(blob, &link)?;
        }
        Ok(link)
    }

    pub fn json(&self, name: &str) -> Result<Value> {
        Ok(serde_json::from_str(&std::fs::read_to_string(self.file(name)?)?)?)
    }
}
