#[cfg(feature = "fresh")]
extern crate bindgen;

#[cfg(feature = "fresh")]
use std::env;
#[cfg(feature = "fresh")]
use std::fs;
#[cfg(feature = "fresh")]
use std::path::{PathBuf};

#[cfg(all(
    not(feature = "fresh"),
    not(any(
        feature = "cuda_6_5",
        feature = "cuda_7_0",
        feature = "cuda_7_5",
        feature = "cuda_8_0",
        feature = "cuda_9_0",
        feature = "cuda_9_1",
        feature = "cuda_9_2",
        feature = "cuda_10_0",
    ))
))]
fn main() {
  compile_error!("a cuda version feature must be enabled");
}

#[cfg(all(
    not(feature = "fresh"),
    any(
        feature = "cuda_6_5",
        feature = "cuda_7_0",
        feature = "cuda_7_5",
        feature = "cuda_8_0",
        feature = "cuda_9_0",
        feature = "cuda_9_1",
        feature = "cuda_9_2",
        feature = "cuda_10_0",
    )
))]
fn main() {
  println!("cargo:rustc-link-lib=cudart");
}

#[cfg(feature = "fresh")]
fn main() {
  println!("cargo:rustc-link-lib=cudart");

  let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
  let cuda_dir = PathBuf::from(
      env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_owned())
  );
  let cuda_include_dir = cuda_dir.join("include");

  #[cfg(feature = "cuda_6_5")]
  let a_cuda_version_feature_must_be_enabled = "v6_5";
  #[cfg(feature = "cuda_7_0")]
  let a_cuda_version_feature_must_be_enabled = "v7_0";
  #[cfg(feature = "cuda_7_5")]
  let a_cuda_version_feature_must_be_enabled = "v7_5";
  #[cfg(feature = "cuda_8_0")]
  let a_cuda_version_feature_must_be_enabled = "v8_0";
  #[cfg(feature = "cuda_9_0")]
  let a_cuda_version_feature_must_be_enabled = "v9_0";
  #[cfg(feature = "cuda_9_1")]
  let a_cuda_version_feature_must_be_enabled = "v9_1";
  #[cfg(feature = "cuda_9_2")]
  let a_cuda_version_feature_must_be_enabled = "v9_2";
  #[cfg(feature = "cuda_10_0")]
  let a_cuda_version_feature_must_be_enabled = "v10_0";
  let v = a_cuda_version_feature_must_be_enabled;

  let gensrc_dir = manifest_dir.join("src").join("ffi").join(v);
  fs::create_dir(&gensrc_dir).ok();

  fs::remove_file(gensrc_dir.join("_cuda_runtime_api.rs")).ok();
  bindgen::Builder::default()
    .clang_arg(format!("-I{}", cuda_include_dir.as_os_str().to_str().unwrap()))
    .header("wrapped_cuda_runtime_api.h")
    .whitelist_recursively(false)
    // Device management.
    .whitelist_function("cudaDeviceReset")
    .whitelist_function("cudaDeviceSynchronize")
    .whitelist_function("cudaGetDeviceCount")
    .whitelist_function("cudaGetDevice")
    .whitelist_function("cudaGetDeviceFlags")
    .whitelist_function("cudaGetDeviceProperties")
    .whitelist_function("cudaDeviceGetAttribute")
    .whitelist_function("cudaSetDevice")
    .whitelist_function("cudaSetDeviceFlags")
    // Error handling.
    .whitelist_function("cudaGetErrorString")
    // Stream management.
    .whitelist_function("cudaStreamCreate")
    .whitelist_function("cudaStreamCreateWithFlags")
    .whitelist_function("cudaStreamCreateWithPriority")
    .whitelist_function("cudaStreamDestroy")
    .whitelist_function("cudaStreamAddCallback")
    .whitelist_function("cudaStreamAttachMemAsync")
    .whitelist_function("cudaStreamQuery")
    .whitelist_function("cudaStreamSynchronize")
    .whitelist_function("cudaStreamWaitEvent")
    // Event management.
    .whitelist_function("cudaEventCreate")
    .whitelist_function("cudaEventCreateWithFlags")
    .whitelist_function("cudaEventDestroy")
    .whitelist_function("cudaEventElapsedTime")
    .whitelist_function("cudaEventQuery")
    .whitelist_function("cudaEventRecord")
    .whitelist_function("cudaEventSynchronize")
    // Memory management.
    .whitelist_function("cudaMalloc")
    .whitelist_function("cudaFree")
    .whitelist_function("cudaMallocHost")
    .whitelist_function("cudaFreeHost")
    .whitelist_function("cudaHostAlloc")
    .whitelist_function("cudaHostGetDevicePointer")
    .whitelist_function("cudaHostGetFlags")
    .whitelist_function("cudaHostRegister")
    .whitelist_function("cudaHostUnregister")
    .whitelist_function("cudaMallocManaged")
    .whitelist_function("cudaMemAdvise")
    .whitelist_function("cudaMemPrefetchAsync")
    .whitelist_function("cudaMemRangeGetAttribute")
    .whitelist_function("cudaMemRangeGetAttributes")
    .whitelist_function("cudaMemcpy")
    .whitelist_function("cudaMemcpyAsync")
    .whitelist_function("cudaMemcpy2D")
    .whitelist_function("cudaMemcpy2DAsync")
    .whitelist_function("cudaMemcpyPeer")
    .whitelist_function("cudaMemcpyPeerAsync")
    .whitelist_function("cudaMemset")
    .whitelist_function("cudaMemsetAsync")
    // Peer device memory access.
    .whitelist_function("cudaDeviceCanAccessPeer")
    .whitelist_function("cudaDeviceDisablePeerAccess")
    .whitelist_function("cudaDeviceEnablePeerAccess")
    // OpenGL interoperability.
    .whitelist_function("cudaGLGetDevices")
    .whitelist_function("cudaGraphicsGLRegisterBuffer")
    .whitelist_function("cudaGraphicsGLRegisterImage")
    // Graphics interoperability.
    .whitelist_function("cudaGraphicsMapResources")
    .whitelist_function("cudaGraphicsResourceGetMappedPointer")
    .whitelist_function("cudaGraphicsResourceSetMapFlags")
    .whitelist_function("cudaGraphicsUnmapResources")
    .whitelist_function("cudaGraphicsUnregisterResource")
    // Version management.
    .whitelist_function("cudaDriverGetVersion")
    .whitelist_function("cudaRuntimeGetVersion")
    .generate()
    .expect("bindgen failed to generate runtime bindings")
    .write_to_file(gensrc_dir.join("_cuda_runtime_api.rs"))
    .expect("bindgen failed to write runtime bindings");
}
