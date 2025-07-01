// fn main(){}
use ocl::builders::ProgramBuilder;
use ocl::{Buffer, Context, Device, DeviceType, MemFlags, Platform, Queue};
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

    // program_con.build(None::<&str>)?;

    let mut msg = [0u8; 16];
    let msg_buffer = Buffer::<u8>::builder().queue(queue.clone()).flags(MemFlags::new().write_only()).len(16).build()?;
        

    let kernel = ocl::Kernel::builder().program(&program_con).name("vector_bytes").queue(queue.clone()).arg(&msg_buffer).build()?;

    unsafe {kernel.cmd().queue(&queue).global_work_size(1).enq()?;}
    msg_buffer.read(&mut msg[..]).enq()?;
    for b in &msg {
        print!("{:02X} ", b);
    }
    println!();

    Ok(())
}
// use ocl::{Platform, Device, Context, Queue, Program, Kernel, Buffer, flags, DeviceType};

// fn main() -> ocl::Result<()> {
//     // OpenCL kernel source
//     let src = r#"
//     __kernel void vector_bytes(__global uchar16 *test) {
//         uint4 vec = {0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F}; 
//         uchar *p = (uchar*)&vec;
//         *test = (uchar16)(*p, *(p+1), *(p+2), *(p+3), *(p+4), *(p+5), 
//             *(p+6), *(p+7), *(p+8), *(p+9), *(p+10), *(p+11), *(p+12), 
//             *(p+13), *(p+14), *(p+15));
//     }
//     "#;

//     // Platform and device selection
//     let platform = Platform::list().into_iter().next().unwrap();
//     let mut devices = Device::list(platform, Some(DeviceType::GPU)).unwrap_or_default();
//     if devices.is_empty() {
//         devices = Device::list(platform, Some(DeviceType::CPU)).unwrap_or_default();
//     }
//     let device = devices.into_iter().next().unwrap();

//     // Context and queue
//     let context = Context::builder().platform(platform).devices(device.clone()).build()?;
//     let queue = Queue::new(&context, device.clone(), None)?;

//     // Output buffer for uchar16 (16 u8s)
//     let mut out = [0u8; 16];
//     let out_buffer = Buffer::<u8>::builder() .queue(queue.clone()).flags(flags::MEM_WRITE_ONLY) .len(16) .build()?;

//     // Build program and kernel
//     let program = Program::builder() .src(src) .devices(device.clone()).build(&context)?;

//     let kernel = Kernel::builder() .program(&program) .name("vector_bytes").queue(queue.clone()) .arg(&out_buffer) .build()?;

//     unsafe { kernel.cmd() .queue(&queue) .global_work_size(1) .enq()?; }

//     out_buffer.read(&mut out[..]).enq()?;

//     print!("Kernel output: ");
//     for b in &out {
//         print!("{:02X} ", b);
//     }
//     println!();

//     Ok(())
// }
