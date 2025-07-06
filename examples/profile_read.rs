use ocl::builders::ProgramBuilder;
use ocl::{ Context, Device, DeviceType,  Platform, Queue};
use std::{fs};
const NUM:usize = 131072;

fn main() -> Result<(), Box<dyn std::error::Error>> {

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
     let mut buffer = [0i8;NUM*16];
    let buffer_cl = ocl::Buffer::<i8>::builder()
        .queue(queue.clone())
        .flags(ocl::flags::MEM_WRITE_ONLY)
        .len(NUM*16)
        .copy_host_slice(&buffer)
        .build()?;

    let kernel = ocl::Kernel::builder()
        .program(&program_con)
        .name("profile_read")
        .queue(queue.clone())
        .arg(&buffer_cl)
        .arg(NUM as i32)
        .global_work_size(1)
        .build()?;

    unsafe {
        kernel.cmd()
            .queue(&queue)
            .global_work_size(1)
            .enq()?;
    }

    buffer_cl.read(&mut buffer[..]).enq()?;

        println!("Output:");
    for i in 0..NUM {
        print!("c[{}]: ", i);
        for j in 0..16 {
            print!("{:3} ", buffer[i * 16 + j]);
        }
        println!();
    }


    Ok(())
}