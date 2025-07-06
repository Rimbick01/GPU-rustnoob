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
    let mut buffer = vec![0.0f32; 4096];
    let buffer_cl = ocl::Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(ocl::flags::MEM_WRITE_ONLY)
        .len(4096)
        .build()?;

    let kernel = ocl::Kernel::builder()
        .program(&program_con)
        .name("callback")
        .queue(queue.clone())
        .arg(&buffer_cl)
        .global_work_size(1)
        .build()?;

    unsafe {
        kernel.cmd()
            .queue(&queue)
            .global_work_size(1)
            .enq()?;
    }

    buffer_cl.read(&mut buffer[..]).enq()?;

    let mut check = true;
    for i in 0..4096 {
        if buffer[i] != 5.0f32 {
            check = false;
            break;
        }
    }
    if check {
            println!("The data has been initialized successfully.");
        } else {
            println!("The data has not been initialized successfully.");
        }

    assert!(buffer.iter().all(|&v| v == 5.0));
    println!("All values are 5.0: OK");

    Ok(())
}