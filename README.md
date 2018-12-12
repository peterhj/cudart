These are Rust wrappers around the [CUDA runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html).

The FFI bindings are done via [bindgen](https://github.com/rust-lang-nursery/rust-bindgen).
Currently, the bindings are substantially whitelisted, i.e. only a small set of
"core" runtime API functionality is exposed.

This originated as a copy-paste of the [cuda crate](https://github.com/peterhj/libcuda),
but with an eye toward future [cuda-sys](https://github.com/rust-cuda/cuda-sys)
integration which may be facilitated by separating the driver and runtime API
wrappers.
