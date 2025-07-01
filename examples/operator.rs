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

    let mut msg = vec! [0; 4];
    let msg_buffer = Buffer::<i32>::builder().queue(queue.clone()).flags(MemFlags::new().write_only()).len(4).build()?;
        

    let kernel = ocl::Kernel::builder().program(&program_con).name("op_test").queue(queue.clone()).arg(&msg_buffer).build()?;

    unsafe {kernel.cmd().queue(&queue).global_work_size(1).enq()?;}
    msg_buffer.read(&mut msg[..]).enq()?;
    println!("Kernel output: {:?}", msg);

    Ok(())
}
/*
__kernel void op_test(__global int4 *output) {

   int4 vec = (int4)(1, 2, 3, 4);
   vec += 4;

   if(vec.s2 == 7)
      vec &= (int4)(-1, -1, 0, -1);
   
   vec.s01 = vec.s23 < 7; 
   
   while(vec.s3 > 7 && (vec.s0 < 16 || vec.s1 < 16))
      vec.s3 >>= 1; 
      
   *output = vec;
}
*/