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

    let program_con = ProgramBuilder::new() .src(&program_handle).devices(dev.clone()) .build(&context)?;
    let mut s1 = [0.0f32; 8];
    let mut s2 = [0u8; 16];

    let s1_buffer = Buffer::<f32>::builder() .queue(queue.clone()).flags(flags::MEM_WRITE_ONLY) .len(8) .build()?;
    let s2_buffer = Buffer::<u8>::builder() .queue(queue.clone()).flags(flags::MEM_WRITE_ONLY) .len(16) .build()?;

    let kernel = ocl::Kernel::builder().program(&program_con).name("shuffle_test").queue(queue.clone()).arg(&s1_buffer).arg(&s2_buffer).global_work_size(1).build()?;

    unsafe {kernel.cmd().queue(&queue) .global_work_size(1) .enq()?; }
    s1_buffer.read(&mut s1[..]).enq()?;
    s2_buffer.read(&mut s2[..]).enq()?;

    println!("s1 (float8): {:?}", s1);
    print!("s2 (char16): ");
    for &c in &s2 {
        print!("{}", c as u8 as char);
    }
    println!();

    Ok(())
}