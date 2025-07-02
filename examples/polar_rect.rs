use ocl::builders::ProgramBuilder;
use ocl::{flags, Buffer, Context, Device, DeviceType, Platform, Queue};
use std::{fs};

const  M_PI:f32 = 3.14159265358979323846;

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

    let r_coords = [2.0f32, 1.0f32, 3.0f32, 4.0f32];
    let angles = [3.0*M_PI/8.0f32, 3.0*M_PI/4.0f32, 4.0*M_PI/3.0f32, 11.0*M_PI/6.0f32];

    let mut x_coords = [0.0f32; 4];
    let mut y_coords = [0.0f32; 4];

    let r_coords_buf = Buffer::<f32>::builder() .queue(queue.clone()).flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR) .len(4).copy_host_slice(&r_coords) .build()?;
    let x_coords_buf = Buffer::<f32>::builder().queue(queue.clone()) .flags(flags::MEM_WRITE_ONLY) .len(4) .build()?;
    let y_coords_buf = Buffer::<f32>::builder() .queue(queue.clone()).flags(flags::MEM_WRITE_ONLY) .len(4) .build()?;
    let angles_buf = Buffer::<f32>::builder().queue(queue.clone()) .flags(flags::MEM_READ_ONLY |flags::MEM_COPY_HOST_PTR) .len(4) .copy_host_slice(&angles) .build()?;

    let kernel = ocl::Kernel::builder()
        .program(&program_con)
        .name("polar_rect")
        .queue(queue.clone())
        .arg(&r_coords_buf)
        .arg(&angles_buf)
        .arg(&x_coords_buf)
        .arg(&y_coords_buf)
        .global_work_size(1)
        .build()?;

    unsafe {
        kernel.cmd()
            .queue(&queue)
            .global_work_size(1)
            .enq()?;
    }

    x_coords_buf.read(&mut x_coords[..]).enq()?;
    y_coords_buf.read(&mut y_coords[..]).enq()?;
    println!("output");
    for i in 0..4 {
        println!("{:?} , {:?}", x_coords[i], y_coords[i]);
    }
    println!();

    Ok(())
}
/*
__kernel void polar_rect(__global float4 *r_vals, 
                         __global float4 *angles,
                         __global float4 *x_coords, 
                         __global float4 *y_coords) {

   *y_coords = sincos(*angles, x_coords);
   *x_coords *= *r_vals;
   *y_coords *= *r_vals;
}
*/