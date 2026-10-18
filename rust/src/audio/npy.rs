//! Minimal NumPy `.npy` / `.npz` reader for MLX weight files.
//!
//! mlx-community Whisper repos ship either `weights.safetensors` (loaded by
//! mlx-rs directly) or `weights.npz` (a zip of `.npy` files, handled here).

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result, bail};
use half::f16;
use mlx_rs::Array;

/// Parse one `.npy` buffer into an MLX array (little-endian f16/f32/i32/i64, C order).
pub fn parse_npy(bytes: &[u8]) -> Result<Array> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        bail!("not an .npy file");
    }
    let (header_len, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        _ => (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12),
    };
    let header = std::str::from_utf8(&bytes[start..start + header_len]).context("npy header")?;
    let data = &bytes[start + header_len..];

    let descr = field(header, "descr").context("npy descr")?;
    let descr = descr.trim_start_matches(|c| c == '\'' || c == '"');
    let descr = &descr[..descr.find(|c| c == '\'' || c == '"').unwrap_or(descr.len())];
    if field(header, "fortran_order").is_some_and(|f| f.starts_with("True")) {
        bail!("fortran-order npy is not supported");
    }
    let shape_str = field(header, "shape").context("npy shape")?;
    let shape: Vec<i32> = shape_str
        .trim_start_matches('(')
        .split(')')
        .next()
        .unwrap_or("")
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<i32>().context("npy shape dim"))
        .collect::<Result<_>>()?;

    let n: usize = shape.iter().map(|&d| d as usize).product();
    let chunks = |w: usize| {
        if data.len() < n * w {
            bail!("npy data too short: {} < {}", data.len(), n * w);
        }
        Ok(data[..n * w].chunks_exact(w))
    };
    Ok(match descr {
        "<f2" => Array::from_slice(&chunks(2)?.map(|c| f16::from_le_bytes([c[0], c[1]])).collect::<Vec<_>>(), &shape),
        "<f4" => Array::from_slice(
            &chunks(4)?.map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>(),
            &shape,
        ),
        "<i4" => Array::from_slice(
            &chunks(4)?.map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>(),
            &shape,
        ),
        "<i8" => Array::from_slice(
            &chunks(8)?.map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>(),
            &shape,
        ),
        other => bail!("unsupported npy dtype {other}"),
    })
}

/// Value of `'key': value` inside the npy header dict.
fn field<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("'{key}':");
    let rest = &header[header.find(&pat)? + pat.len()..];
    Some(rest.trim_start())
}

/// Load every array in a `.npz` archive, keyed by entry name without `.npy`.
pub fn load_npz(path: &Path) -> Result<HashMap<String, Array>> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut zip = zip::ZipArchive::new(file).context("read npz")?;
    let mut out = HashMap::new();
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)?;
        let name = entry.name()?.trim_end_matches(".npy").to_string();
        let mut buf = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut buf)?;
        let arr = parse_npy(&buf).with_context(|| format!("parse {name}"))?;
        out.insert(name, arr);
    }
    Ok(out)
}
