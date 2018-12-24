#[cfg(not(feature = "cuda_sys"))]
use crate::ffi::driver_types::*;
#[cfg(not(feature = "cuda_sys"))]
use crate::ffi::runtime::*;

#[cfg(feature = "cuda_sys")]
use cuda_sys::cudart::*;

use std::ffi::{CStr};
use std::mem::{size_of, zeroed};
use std::os::raw::{c_void, c_int, c_uint};
use std::ptr::{null_mut};

#[cfg(feature = "cuda_sys")]
const cudaError_cudaSuccess: cudaError_t = cudaError_t::Success;
#[cfg(feature = "cuda_sys")]
const cudaError_cudaErrorPeerAccessAlreadyEnabled: cudaError_t = cudaError_t::PeerAccessAlreadyEnabled;
#[cfg(feature = "cuda_sys")]
const cudaError_cudaErrorPeerAccessNotEnabled: cudaError_t = cudaError_t::PeerAccessNotEnabled;
#[cfg(feature = "cuda_sys")]
const cudaError_cudaErrorCudartUnloading: cudaError_t = cudaError_t::CudartUnloading;
#[cfg(feature = "cuda_sys")]
const cudaError_cudaErrorNotReady: cudaError_t = cudaError_t::NotReady;

#[derive(Clone, Copy, Debug)]
pub struct CudaError(pub cudaError_t);

impl CudaError {
  pub fn get_code(&self) -> u32 {
    let &CudaError(e) = self;
    e as _
  }

  pub fn get_string(&self) -> String {
    let raw_s = unsafe { cudaGetErrorString(self.0) };
    if raw_s.is_null() {
      return format!("(null)");
    }
    let cs = unsafe { CStr::from_ptr(raw_s) };
    let s = match cs.to_str() {
      Err(_) => "(invalid utf8)",
      Ok(s) => s,
    };
    s.to_owned()
  }
}

pub type CudaResult<T> = Result<T, CudaError>;

pub struct CudaDevice;

impl CudaDevice {
  /// Count the number of devices.
  ///
  /// Corresponds to `cudaGetDeviceCount`.
  pub fn count() -> CudaResult<usize> {
    let mut count: c_int = 0;
    match unsafe { cudaGetDeviceCount(&mut count as *mut c_int) } {
      cudaError_cudaSuccess => {
        assert!(count >= 0);
        Ok(count as usize)
      }
      e => Err(CudaError(e)),
    }
  }

  /// Query the current device.
  ///
  /// Corresponds to `cudaGetDevice`.
  pub fn get_current() -> CudaResult<i32> {
    let mut curr_dev: c_int = 0;
    match unsafe { cudaGetDevice(&mut curr_dev as *mut c_int) } {
      cudaError_cudaSuccess => Ok(curr_dev),
      e => Err(CudaError(e)),
    }
  }

  /// Set the current device.
  ///
  /// Corresponds to `cudaSetDevice`.
  pub fn set_current(this_dev: i32) -> CudaResult<()> {
    match unsafe { cudaSetDevice(this_dev as c_int) } {
      cudaError_cudaSuccess => Ok(()),
      e => Err(CudaError(e)),
    }
  }

  /// Reset the current device.
  ///
  /// Corresponds to `cudaDeviceReset`.
  pub fn reset() -> CudaResult<()> {
    match unsafe { cudaDeviceReset() } {
      cudaError_cudaSuccess => Ok(()),
      e => Err(CudaError(e)),
    }
  }

  /// Synchronize all work on the current device.
  ///
  /// Corresponds to `cudaDeviceSynchronize`.
  pub fn synchronize() -> CudaResult<()> {
    match unsafe { cudaDeviceSynchronize() } {
      cudaError_cudaSuccess => Ok(()),
      e => Err(CudaError(e)),
    }
  }

  /// Set flags for the current device.
  ///
  /// Corresponds to `cudaSetDeviceFlags`.
  pub fn set_flags(flags: u32) -> CudaResult<()> {
    match unsafe { cudaSetDeviceFlags(flags as c_uint) } {
      cudaError_cudaSuccess => Ok(()),
      e => Err(CudaError(e)),
    }
  }

  /// Query the `cudaDeviceProp` properties struct for the given device.
  ///
  /// Corresponds to `cudaGetDeviceProperties`.
  pub fn get_properties(this_dev: i32) -> CudaResult<cudaDeviceProp> {
    let mut prop: cudaDeviceProp = unsafe { zeroed() };
    match unsafe { cudaGetDeviceProperties(&mut prop as *mut cudaDeviceProp, this_dev as c_int) } {
      cudaError_cudaSuccess => Ok(prop),
      e => Err(CudaError(e)),
    }
  }

  /// Query the given attribute for the given device.
  ///
  /// Corresponds to `cudaGetDeviceAttribute`.
  pub fn get_attribute(this_dev: i32, attr: cudaDeviceAttr) -> CudaResult<i32> {
    let mut value: c_int = 0;
    match unsafe { cudaDeviceGetAttribute(&mut value as *mut c_int, attr, this_dev as c_int) } {
      cudaError_cudaSuccess => Ok(value as i32),
      e => Err(CudaError(e)),
    }
  }

  /// Check whether peer device access from `this_dev` to `peer_dev` can be
  /// enabled.
  ///
  /// Corresponds to `cudaDeviceCanAccessPeer`.
  pub fn can_access_peer(this_dev: i32, peer_dev: i32) -> CudaResult<bool> {
    let mut access: c_int = 0;
    match unsafe { cudaDeviceCanAccessPeer(&mut access as *mut c_int, this_dev as c_int, peer_dev as c_int) } {
      cudaError_cudaSuccess => Ok(access != 0),
      e => Err(CudaError(e)),
    }
  }

  /// Enable peer device access from the current device to `peer_dev`.
  /// Returns whether or not peer device access was previously enabled.
  ///
  /// Corresponds to `cudaDeviceEnablePeerAccess`.
  pub fn enable_peer_access(peer_dev: i32) -> CudaResult<bool> {
    match unsafe { cudaDeviceEnablePeerAccess(peer_dev as c_int, 0) } {
      cudaError_cudaSuccess => Ok(false),
      cudaError_cudaErrorPeerAccessAlreadyEnabled => Ok(true),
      e => Err(CudaError(e)),
    }
  }

  /// Disable peer device access from the current device to `peer_dev`.
  /// Returns whether or not peer device access was previously enabled.
  ///
  /// Corresponds to `cudaDeviceDisablePeerAccess`.
  pub fn disable_peer_access(peer_dev: i32) -> CudaResult<bool> {
    match unsafe { cudaDeviceDisablePeerAccess(peer_dev as c_int) } {
      cudaError_cudaSuccess => Ok(true),
      cudaError_cudaErrorPeerAccessNotEnabled => Ok(false),
      e => Err(CudaError(e)),
    }
  }
}

#[derive(Debug)]
pub struct CudaStream {
  ptr:  cudaStream_t,
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl Drop for CudaStream {
  fn drop(&mut self) {
    if !self.ptr.is_null() {
      match unsafe { cudaStreamDestroy(self.ptr) } {
        cudaError_cudaSuccess => {}
        cudaError_cudaErrorCudartUnloading => {
          // NB(20160308): Sometimes drop() is called while the global runtime
          // is shutting down; suppress these errors.
        }
        e => {
          let err = CudaError(e);
          panic!("FATAL: CudaStream::drop() failed: {:?} ({})",
              err, err.get_string());
        }
      }
    }
  }
}

impl CudaStream {
  pub fn default() -> CudaStream {
    CudaStream{ptr: null_mut()}
  }

  pub fn create() -> CudaResult<CudaStream> {
    let mut ptr: cudaStream_t = null_mut();
    match unsafe { cudaStreamCreate(&mut ptr as *mut cudaStream_t) } {
      cudaError_cudaSuccess => Ok(CudaStream{ptr: ptr}),
      e => Err(CudaError(e)),
    }
  }

  pub unsafe fn as_mut_ptr(&mut self) -> cudaStream_t {
    self.ptr
  }

  pub fn ptr_eq(&self, other: &CudaStream) -> bool {
    self.ptr == other.ptr
  }

  pub fn add_callback(&mut self, callback: extern "C" fn (stream: cudaStream_t, status: cudaError_t, user_data: *mut c_void), user_data: *mut c_void) -> CudaResult<()> {
    match unsafe { cudaStreamAddCallback(self.ptr, Some(callback), user_data, 0) } {
      cudaError_cudaSuccess => Ok(()),
      e => Err(CudaError(e)),
    }
  }

  pub fn synchronize(&mut self) -> CudaResult<()> {
    match unsafe { cudaStreamSynchronize(self.ptr) } {
      cudaError_cudaSuccess => Ok(()),
      e => Err(CudaError(e)),
    }
  }

  pub fn wait_event(&mut self, event: &mut CudaEvent) -> CudaResult<()> {
    match unsafe { cudaStreamWaitEvent(self.ptr, event.as_mut_ptr(), 0) } {
      cudaError_cudaSuccess => Ok(()),
      e => Err(CudaError(e))
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub enum CudaEventStatus {
  Complete,
  NotReady,
}

#[derive(Debug)]
pub struct CudaEvent {
  ptr:  cudaEvent_t,
}

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl Drop for CudaEvent {
  fn drop(&mut self) {
    if !self.ptr.is_null() {
      match unsafe { cudaEventDestroy(self.ptr) } {
        cudaError_cudaSuccess => {}
        cudaError_cudaErrorCudartUnloading => {
          // NB(20160308): Sometimes drop() is called while the global runtime
          // is shutting down; suppress these errors.
        }
        e => {
          let err = CudaError(e);
          panic!("FATAL: CudaEvent::drop() failed: {:?} ({})",
              err, err.get_string());
        }
      }
    }
  }
}

impl CudaEvent {
  pub fn create() -> CudaResult<CudaEvent> {
    let mut ptr = null_mut() as cudaEvent_t;
    match unsafe { cudaEventCreate(&mut ptr as *mut cudaEvent_t) } {
      cudaError_cudaSuccess => Ok(CudaEvent{ptr: ptr}),
      e => Err(CudaError(e)),
    }
  }

  pub fn create_blocking() -> CudaResult<CudaEvent> {
    Self::create_with_flags(0x01)
  }

  pub fn create_fastest() -> CudaResult<CudaEvent> {
    Self::create_with_flags(0x02)
  }

  pub fn create_with_flags(flags: u32) -> CudaResult<CudaEvent> {
    let mut ptr = null_mut() as cudaEvent_t;
    match unsafe { cudaEventCreateWithFlags(&mut ptr as *mut cudaEvent_t, flags) } {
      cudaError_cudaSuccess => Ok(CudaEvent{ptr: ptr}),
      e => Err(CudaError(e)),
    }
  }

  pub unsafe fn as_mut_ptr(&mut self) -> cudaEvent_t {
    self.ptr
  }

  pub fn ptr_eq(&self, other: &CudaEvent) -> bool {
    self.ptr == other.ptr
  }

  pub fn query(&mut self) -> CudaResult<CudaEventStatus> {
    match unsafe { cudaEventQuery(self.ptr) } {
      cudaError_cudaSuccess => Ok(CudaEventStatus::Complete),
      cudaError_cudaErrorNotReady => Ok(CudaEventStatus::NotReady),
      e => Err(CudaError(e)),
    }
  }

  pub fn record(&mut self, stream: &mut CudaStream) -> CudaResult<()> {
    match unsafe { cudaEventRecord(self.ptr, stream.as_mut_ptr()) } {
      cudaError_cudaSuccess => Ok(()),
      e => Err(CudaError(e)),
    }
  }

  pub fn synchronize(&mut self) -> CudaResult<()> {
    match unsafe { cudaEventSynchronize(self.ptr) } {
      cudaError_cudaSuccess => Ok(()),
      e => Err(CudaError(e)),
    }
  }
}

pub fn cuda_alloc_device(size: usize) -> CudaResult<*mut u8> {
  let mut dptr: *mut c_void = null_mut();
  match unsafe { cudaMalloc(&mut dptr as *mut *mut c_void, size) } {
    cudaError_cudaSuccess => Ok(dptr as *mut u8),
    e => Err(CudaError(e)),
  }
}

pub fn cuda_alloc_host(size: usize) -> CudaResult<*mut u8> {
  let mut ptr: *mut c_void = null_mut();
  match unsafe { cudaMallocHost(&mut ptr as *mut *mut c_void, size) } {
    cudaError_cudaSuccess => Ok(ptr as *mut u8),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_free_device(dptr: *mut u8) -> CudaResult<()> {
  match cudaFree(dptr as *mut c_void) {
    cudaError_cudaSuccess => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_free_host(ptr: *mut u8) -> CudaResult<()> {
  match cudaFreeHost(ptr as *mut c_void) {
    cudaError_cudaSuccess => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_memset(dptr: *mut u8, value: i32, size: usize) -> CudaResult<()> {
  match cudaMemset(dptr as *mut c_void, value, size) {
    cudaError_cudaSuccess => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_memset_async(dptr: *mut u8, value: i32, size: usize, stream: &mut CudaStream) -> CudaResult<()> {
  match cudaMemsetAsync(dptr as *mut c_void, value, size, stream.as_mut_ptr()) {
    cudaError_cudaSuccess => Ok(()),
    e => Err(CudaError(e)),
  }
}

#[derive(Clone, Copy, Debug)]
pub enum CudaMemcpyKind {
  HostToHost,
  HostToDevice,
  DeviceToHost,
  DeviceToDevice,
  Unified,
}

impl CudaMemcpyKind {
  #[cfg(not(feature = "cuda_sys"))]
  pub fn to_raw(&self) -> cudaMemcpyKind {
    match *self {
      CudaMemcpyKind::HostToHost      => cudaMemcpyKind_cudaMemcpyHostToHost,
      CudaMemcpyKind::HostToDevice    => cudaMemcpyKind_cudaMemcpyHostToDevice,
      CudaMemcpyKind::DeviceToHost    => cudaMemcpyKind_cudaMemcpyDeviceToHost,
      CudaMemcpyKind::DeviceToDevice  => cudaMemcpyKind_cudaMemcpyDeviceToDevice,
      CudaMemcpyKind::Unified         => cudaMemcpyKind_cudaMemcpyDefault,
    }
  }
}

pub unsafe fn cuda_memcpy<T>(
    dst: *mut T,
    src: *const T,
    len: usize,
    kind: CudaMemcpyKind) -> CudaResult<()>
where T: Copy + 'static
{
  match cudaMemcpy(
      dst as *mut c_void,
      src as *const c_void,
      len * size_of::<T>(),
      kind.to_raw())
  {
    cudaError_cudaSuccess => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_memcpy_async<T>(
    dst: *mut T,
    src: *const T,
    len: usize,
    kind: CudaMemcpyKind,
    stream: &mut CudaStream) -> CudaResult<()>
where T: Copy + 'static
{
  match cudaMemcpyAsync(
      dst as *mut c_void,
      src as *const c_void,
      len * size_of::<T>(),
      kind.to_raw(),
      stream.as_mut_ptr())
  {
    cudaError_cudaSuccess => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_memcpy_2d_async<T>(
    dst: *mut T,
    dst_pitch_bytes: usize,
    src: *const T,
    src_pitch_bytes: usize,
    width: usize,
    height: usize,
    kind: CudaMemcpyKind,
    stream: &mut CudaStream) -> CudaResult<()>
where T: Copy + 'static
{
  let width_bytes = width * size_of::<T>();
  assert!(width_bytes <= dst_pitch_bytes);
  assert!(width_bytes <= src_pitch_bytes);
  match cudaMemcpy2DAsync(
      dst as *mut c_void,
      dst_pitch_bytes,
      src as *const c_void,
      src_pitch_bytes,
      width_bytes,
      height,
      kind.to_raw(),
      stream.as_mut_ptr())
  {
    cudaError_cudaSuccess => Ok(()),
    e => Err(CudaError(e)),
  }
}

pub unsafe fn cuda_memcpy_peer_async<T>(
    dst: *mut T,
    dst_device_idx: i32,
    src: *const T,
    src_device_idx: i32,
    len: usize,
    stream: &mut CudaStream) -> CudaResult<()>
where T: Copy + 'static
{
  match cudaMemcpyPeerAsync(
      dst as *mut c_void,
      dst_device_idx,
      src as *const c_void,
      src_device_idx,
      len * size_of::<T>(),
      stream.as_mut_ptr())
  {
    cudaError_cudaSuccess => Ok(()),
    e => Err(CudaError(e)),
  }
}
