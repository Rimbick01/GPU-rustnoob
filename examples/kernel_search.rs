use cl_sys::{self as cl, libc};
use libc::c_void;
use std::ffi::CStr;
use std::ptr;

const PROGRAM_FILE: &str = "test.cl";

fn main() {
    let mut platform: cl::cl_platform_id = ptr::null_mut();
    let mut err;

    err = unsafe { cl::clGetPlatformIDs(1, &mut platform, ptr::null_mut()) };
    if err != cl::CL_SUCCESS {
        panic!("Couldn't find any platforms: {}", err);
    }

    let mut device: cl::cl_device_id = ptr::null_mut();
    err = unsafe {cl::clGetDeviceIDs(platform,cl::CL_DEVICE_TYPE_GPU,1,&mut device,ptr::null_mut(),)};
    if err == cl::CL_DEVICE_NOT_FOUND {
        err = unsafe {cl::clGetDeviceIDs(platform,cl::CL_DEVICE_TYPE_CPU,1,&mut device,ptr::null_mut(),)};
    }
    if err != cl::CL_SUCCESS {
        panic!("Couldn't find any devices: {}", err);
    }

    // let mut context: cl::cl_context = ptr::null_mut();
    let context: cl::cl_context ;
    let mut err = cl::CL_SUCCESS;
    context = unsafe {cl::clCreateContext(ptr::null(),1,&device as *const cl::cl_device_id,None,ptr::null_mut(),&mut err,)};
    if err != cl::CL_SUCCESS {
        panic!("Couldn't create context: {}", err);
    }

    let program_source = match std::fs::read(PROGRAM_FILE) {
        Ok(content) => content,
        Err(e) => panic!("Couldn't read program file: {}", e),
    };

    let program: cl::cl_program ;
    let strings: [*const libc::c_char; 1] = [program_source.as_ptr() as *const _];
    let lengths: [usize; 1] = [program_source.len()];
    program = unsafe {
        cl::clCreateProgramWithSource(context, 1, strings.as_ptr(), lengths.as_ptr(), &mut err)
    };
    if err != cl::CL_SUCCESS {
        panic!("Couldn't create program: {}", err);
    }

    err =
        unsafe { cl::clBuildProgram(program, 0, ptr::null(), ptr::null(), None, ptr::null_mut()) };
    if err != cl::CL_SUCCESS {
        let mut log_size = 0;
        unsafe {cl::clGetProgramBuildInfo(program,device,cl::CL_PROGRAM_BUILD_LOG,0,ptr::null_mut(),&mut log_size,)};
        let mut program_log = vec![0u8; log_size];
        unsafe {cl::clGetProgramBuildInfo(program,device,cl::CL_PROGRAM_BUILD_LOG,log_size,program_log.as_mut_ptr() as *mut c_void,ptr::null_mut(),)};
        let log_cstr =
            CStr::from_bytes_with_nul(&program_log).expect("Build log is not null-terminated");
        println!("Build log:\n{}", log_cstr.to_str().unwrap());
        panic!("Program build failed: {}", err);
    }

    let mut num_kernels = 0;
    err = unsafe { cl::clCreateKernelsInProgram(program, 0, ptr::null_mut(), &mut num_kernels) };
    if err != cl::CL_SUCCESS {
        panic!("Couldn't get number of kernels: {}", err);
    }

    let mut kernels = vec![ptr::null_mut(); num_kernels as usize];
    err = unsafe {cl::clCreateKernelsInProgram(program, num_kernels, kernels.as_mut_ptr(), ptr::null_mut())};
    if err != cl::CL_SUCCESS {
        panic!("Couldn't create kernels: {}", err);
    }

    for (i, &kernel) in kernels.iter().enumerate() {
        if kernel.is_null() {
            continue;
        }

        let mut name_len = 0;
        err = unsafe {cl::clGetKernelInfo(kernel,cl::CL_KERNEL_FUNCTION_NAME,0,ptr::null_mut(),&mut name_len,)};
        if err != cl::CL_SUCCESS {
            panic!("Couldn't get kernel name length: {}", err);
        }

        let mut name_buf = vec![0u8; name_len];
        err = unsafe {cl::clGetKernelInfo(kernel,cl::CL_KERNEL_FUNCTION_NAME,name_len,name_buf.as_mut_ptr() as *mut c_void,ptr::null_mut(),)};
        if err != cl::CL_SUCCESS {
            panic!("Couldn't get kernel name: {}", err);
        }

        let name = CStr::from_bytes_with_nul(&name_buf).expect("Invalid kernel name").to_str().expect("Invalid UTF-8 in kernel name");

        if name == "mult" {
            println!("Found mult kernel at index {}", i);
            break;
        }
    }

    for &kernel in &kernels {
        if !kernel.is_null() {unsafe { cl::clReleaseKernel(kernel) };}
    }
    unsafe {
        cl::clReleaseProgram(program);
        cl::clReleaseContext(context);
    }
}
/* 
__kernel void add(__global float *a,__global float *b,__global float *c) {
    int gid = get_global_id(0);
    c[gid] = a[gid] + b[gid];
}

__kernel void sub(__global float *a,__global float *b,__global float *c) {   
    int gid = get_global_id(0);
    c[gid] = a[gid] - b[gid];
}

__kernel void mult(__global float *a,__global float *b,__global float *c) {   
    int gid = get_global_id(0);
    c[gid] = a[gid] * b[gid];
}

__kernel void div(__global float *a,__global float *b,__global float *c) {   
    int gid = get_global_id(0);
    c[gid] = a[gid] / b[gid];
}
    */