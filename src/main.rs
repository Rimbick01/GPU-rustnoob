// fn main(){}
use ocl::builders::ProgramBuilder;
use ocl::{ Context, Device, DeviceType,  Platform, Queue};
use std::{fs};

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
    let num_items = 2;
    let num_vectors = 4;
    let num_ints = num_vectors * 4;
    let mut x = [0i32; 16];
    // Fill with 0..15 for demonstration
    for i in 0..16 {
        x[i] = i as i32;
    }

    let x_buffer = ocl::Buffer::<i32>::builder()
    .queue(queue.clone())
    .flags(ocl::flags::MEM_READ_WRITE | ocl::flags::MEM_COPY_HOST_PTR)
    .len(num_ints)
    .copy_host_slice(&x)
    .build()?;


    let kernel = ocl::Kernel::builder()
        .program(&program_con)
        .name("profile_items")
        .queue(queue.clone())
        .arg(&x_buffer)
        .arg(num_ints as i32)
        .global_work_size(num_items)
        .build()?;

    unsafe {
        kernel.cmd()
            .queue(&queue)
            .global_work_size(num_items)
            .enq()?;
    }

    x_buffer.read(&mut x[..]).enq()?;

    // Print the output as 4 int4 vectors
    println!("Output:");
    for i in 0..num_vectors {
        print!("x[{}]: ", i);
        for j in 0..4 {
            print!("{:4} ", x[i * 4 + j]);
        }
        println!();
    }


    Ok(())
}
