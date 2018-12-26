#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

pub mod cuda_runtime_api {
use cuda_ffi_types::cuda_runtime_api::*;
use cuda_ffi_types::driver_types::*;
include!(concat!(env!("OUT_DIR"), "/cuda_runtime_api.rs"));
}
