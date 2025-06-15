use ocl::ffi::cl_context;
use ocl::{Device, Context, Platform, DeviceType};
use ocl::enums::ContextInfo;
use std::mem;

fn main() {
    let platforms = Platform::list();

    let platform = platforms.into_iter().next().expect("No platforms found");

    let mut devices = Device::list(platform, Some(DeviceType::GPU)).unwrap_or_default();
    if devices.is_empty() {
        devices = Device::list(platform, Some(DeviceType::CPU)).unwrap_or_default();
    }
    let dev = devices.into_iter().next().expect("No devices found");

    let context = Context::builder()
        .devices(dev)
        .build()
        .expect("Failed to create context");

    let cl_ctx: cl_context =  { context.as_core().as_ptr() };

    std::mem::forget(context);

    let get_ref_count = || {
        let mut count = 0u32;
        let mut size = 0;
        unsafe {
            ocl::ffi::clGetContextInfo(
                cl_ctx,
                ContextInfo::ReferenceCount as u32,
                mem::size_of_val(&count),
                &mut count as *mut _ as *mut _,
                &mut size,
            )
        };
        count
    };

    println!("Initial reference count: {}", get_ref_count());
    unsafe { ocl::ffi::clRetainContext(cl_ctx) };
    println!("Reference count after retain: {}", get_ref_count());
    unsafe { ocl::ffi::clReleaseContext(cl_ctx) };
    println!("Reference count after first release: {}", get_ref_count());
    unsafe { ocl::ffi::clReleaseContext(cl_ctx) };
}
