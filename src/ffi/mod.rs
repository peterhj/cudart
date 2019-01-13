#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

pub use self::v::cuda_runtime_api::*;

#[cfg(feature = "cuda_8_0")]
mod v {
  pub mod cuda_runtime_api {
    use cuda_api_types::cuda_runtime_api::*;
    use cuda_api_types::driver_types::*;
    include!("v8_0/_cuda_runtime_api.rs");
  }
}

#[cfg(feature = "cuda_9_0")]
mod v {
  pub mod cuda_runtime_api {
    use cuda_api_types::cuda_runtime_api::*;
    use cuda_api_types::driver_types::*;
    include!("v9_0/_cuda_runtime_api.rs");
  }
}

#[cfg(feature = "cuda_9_2")]
mod v {
  pub mod cuda_runtime_api {
    use cuda_api_types::cuda_runtime_api::*;
    use cuda_api_types::driver_types::*;
    include!("v9_2/_cuda_runtime_api.rs");
  }
}

#[cfg(feature = "cuda_10_0")]
mod v {
  pub mod cuda_runtime_api {
    use cuda_api_types::cuda_runtime_api::*;
    use cuda_api_types::driver_types::*;
    include!("v10_0/_cuda_runtime_api.rs");
  }
}
