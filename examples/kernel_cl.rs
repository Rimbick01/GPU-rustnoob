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
        .build(&context)
        .unwrap(); 
    let mut msg = [0u8; 16];
    let msg_buffer = Buffer::<u8>::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().write_only())
        .len(16)
        .copy_host_slice(&msg) 
        .build()?;
    let kernel = ocl::Kernel::builder()
        .program(&program_con)
        .name("hello_kernel")
        .queue(queue.clone())
        .arg_named("msg", Some(&msg_buffer))
        .build()?;

    kernel.set_arg(0, &msg_buffer)?;
    unsafe {
        kernel.cmd()
            .queue(&queue)
            .global_work_size(1)
            .enq()?;
    }

    msg_buffer.read(&mut msg[..]).enq()?;
    let msg_str = String::from_utf8_lossy(&msg);
    println!("Kernel output: {}", msg_str);

    Ok(())
}
/*
__kernel void hello_kernel(__global char16 *msg) {
    *msg = (char16)(
        'h', 'e', 'l', 'l', 'o', ' ',
        'k', 'e', 'r', 'n', 'e', 'l', '!', '!', '!', '\0');
}
 */