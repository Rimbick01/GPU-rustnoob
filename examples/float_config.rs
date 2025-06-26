use cl_sys::*;
use std::mem;
use std::ptr;

fn main() {
    unsafe {
        let mut platform: cl_platform_id = ptr::null_mut();
        clGetPlatformIDs(1, &mut platform, ptr::null_mut());
        let mut device: cl_device_id = ptr::null_mut();
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &mut device, ptr::null_mut());
        let mut flag: cl_device_fp_config = 0;
        clGetDeviceInfo(
            device,
            CL_DEVICE_SINGLE_FP_CONFIG,
            mem::size_of::<cl_device_fp_config>(),
            &mut flag as *mut _ as *mut _,
            ptr::null_mut(),
        );
        println!("Float processing features:");
        if flag & CL_FP_INF_NAN != 0 {
            println!("INF and NaN values supported.");
        }
        if flag & CL_FP_DENORM != 0 {
            println!("Denormalized numbers supported.");
        }
        if flag & CL_FP_ROUND_TO_NEAREST != 0 {
            println!("Round to nearest even mode supported.");
        }
        if flag & CL_FP_ROUND_TO_INF != 0 {
            println!("Round to infinity mode supported.");
        }
        if flag & CL_FP_ROUND_TO_ZERO != 0 {
            println!("Round to zero mode supported.");
        }
        if flag & CL_FP_SOFT_FLOAT != 0 {
            println!("Floating-point multiply-and-add operation supported.");
        }
    }
}
