//! Qwen3.5 and Whisper on Apple Silicon through MLX (mlx-rs), as a library for the `qwen35`
//! runner, the `audio` example and the Jarvis voice assistant (../jarvis).

pub mod audio;
pub mod hf;
pub mod kernel;
pub mod llm;
pub mod model;

/// Run `f` on this thread with MLX ready for it. MLX keeps its default streams per thread, so a
/// thread that evaluates arrays (Whisper, Qwen) must register streams first or MLX fails with
/// "There is no Stream(cpu, 0) in current thread".
pub fn with_mlx_stream<T>(f: impl FnOnce() -> T) -> T {
    init_mlx_thread();
    f()
}

/// Register a default CPU and GPU stream for the calling thread (idempotent).
pub fn init_mlx_thread() {
    thread_local! { static DONE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
    if DONE.with(|d| d.replace(true)) {
        return;
    }
    // SAFETY: plain mlx-c calls; the scheduler copies the stream, so the handles can be freed.
    unsafe {
        for ty in [mlx_sys::mlx_device_type__MLX_CPU, mlx_sys::mlx_device_type__MLX_GPU] {
            let dev = mlx_sys::mlx_device_new_type(ty, 0);
            let stream = mlx_sys::mlx_stream_new_device(dev);
            mlx_sys::mlx_set_default_stream(stream);
            mlx_sys::mlx_stream_free(stream);
            mlx_sys::mlx_device_free(dev);
        }
    }
}
