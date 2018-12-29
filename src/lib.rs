#![allow(non_upper_case_globals)]

#[cfg(not(feature = "cuda_sys"))]
extern crate cuda_ffi_types;
#[cfg(feature = "cuda_sys")]
extern crate cuda_sys;
#[macro_use] extern crate static_assertions;

pub use crate::runtime::{
  CudaError,
  CudaResult,
  CudaDevice,
  CudaStream,
  CudaEvent,
  CudaEventStatus,
  CudaMemcpyKind,
  cuda_alloc_device,
  cuda_alloc_host,
  cuda_free_device,
  cuda_free_host,
  cuda_memset,
  cuda_memset_async,
  cuda_memcpy,
  cuda_memcpy_async,
  cuda_memcpy_2d_async,
  cuda_memcpy_peer_async,
  get_driver_version,
  get_runtime_version,
};

#[cfg(not(feature = "cuda_sys"))]
pub mod ffi;
pub mod runtime;

#[cfg(feature = "cuda_sys")]
mod version_checks {
  #[cfg(feature = "cuda_8_0")]  const_assert_eq!(cuda_api_version; cuda_sys::cuda::__CUDA_API_VERSION,  8000);
  #[cfg(feature = "cuda_8_0")]  const_assert_eq!(cuda_version;     cuda_sys::cuda::CUDA_VERSION,        8000);
}
