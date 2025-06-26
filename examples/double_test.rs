use ocl::builders::ProgramBuilder;
use ocl::{Buffer, Context, Device, DeviceType, MemFlags, Platform, Queue};
use std::fs;

fn main() -> ocl::Result<()> {

    let platforms = Platform::list();

    let platform = platforms.into_iter().next().expect("No platforms found");

    let mut devices = Device::list(platform, Some(DeviceType::GPU)).unwrap_or_default();
    if devices.is_empty() {
        devices = Device::list(platform, Some(DeviceType::CPU)).unwrap_or_default();
    }
    let dev = devices.into_iter().next().expect("No devices found");

    let context = Context::builder().platform(platform).devices(dev).build()?;
    let program_handle = fs::read_to_string("hello_kernel.cl").unwrap_or_else(|_| panic!("Failed to read file: hello_kernel.cl"));
    let queue = Queue::new(&context, (*dev).into(), None)?;

    let program_con = ProgramBuilder::new()
        .src(&program_handle)
        .devices(dev.clone())
        .build(&context)?;
    // program_con.build(Some("-D FP_64"))?;
    // program_con.build(None::<&str>)?;

    let mut out = [0.0f32; 1];
    let out_buffer = Buffer::<f32>::builder().queue(queue.clone()).flags(MemFlags::new().write_only()).len(1).build()?;
    let a: f32 = 6.0;
    let b: f32 = 5.0;
    let kernel = ocl::Kernel::builder().program(&program_con).name("double_test").queue(queue.clone()).arg(a).arg(b).arg(&out_buffer).build()?;

    unsafe {kernel.cmd().queue(&queue).global_work_size(1).enq()?;}
    out_buffer.read(&mut out[..]).enq()?;
    println!("Kernel output: {:?}", out);

    Ok(())
}
/*
#ifdef FP_64
#pragma OPENCL_EXTENSION cl_khr_fp64 : enable
#endif

__kernel void double_test(
        float a, float b,
        __global float* out) {
#ifdef FP_64
    double c = (double)(a / b);
    *out = c;
#else
    *out = a * b;
#endif
}
 */