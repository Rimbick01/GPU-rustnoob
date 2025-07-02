
use ocl::builders::ProgramBuilder;
use ocl::{flags, Buffer, Context, Device, DeviceType, Platform, Queue};
use std::{fs};

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

    let mut result = [0u32; 2];
    let result_buf = Buffer::<u32>::builder().queue(queue.clone()).flags(flags::MEM_WRITE_ONLY).len(2).build()?;
    let kernel = ocl::Kernel::builder().program(&program_con).name("mad_test").queue(queue.clone()).arg(&result_buf).global_work_size(1).build()?;

    unsafe {kernel.cmd() .queue(&queue) .global_work_size(1) .enq()?; }
    result_buf.read(&mut result[..]).enq()?;
    println!("{:?} , {:?}", result[1], result[0]);
    println!("mad24(a, b, c) = 0x{:X} ({:?})", result[0], result[0]);
    println!("mad_hi(a, b, c) = 0x{:X} ({:?})", result[1], result[1]);
    println!();

    Ok(())
}