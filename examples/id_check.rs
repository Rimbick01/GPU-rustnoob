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

    let global_offset = [3, 5];
    let global_size = [6, 4];
    let local_size = [3, 2];
    let output_len = global_size[0] * global_size[1];

    let mut msg = vec![0.0f32; output_len];
    let msg_buffer = Buffer::<f32>::builder().queue(queue.clone()).flags(MemFlags::new().write_only()).len(output_len).build()?;
        

    let kernel = ocl::Kernel::builder().program(&program_con).name("id_check").queue(queue.clone()).arg(&msg_buffer).global_work_size(global_size).build()?;

    unsafe {kernel.cmd().queue(&queue).global_work_size(global_size).global_work_offset(global_offset).local_work_size(local_size).enq()?;}
    msg_buffer.read(&mut msg).enq()?;
    for i in 0..global_size[1] {
        for j in 0..global_size[0] {
            print!("{:6.2}", msg[i * global_size[0] + j]);
        }
        println!();
    }

    Ok(())
}
/*
__kernel void id_check(__global float *output) { 

   size_t global_id_0 = get_global_id(0);
   size_t global_id_1 = get_global_id(1);
   size_t global_size_0 = get_global_size(0);
   size_t offset_0 = get_global_offset(0);
   size_t offset_1 = get_global_offset(1);
   size_t local_id_0 = get_local_id(0);
   size_t local_id_1 = get_local_id(1);

   int index_0 = global_id_0 - offset_0;
   int index_1 = global_id_1 - offset_1;
   int index = index_1 * global_size_0 + index_0;
   
   float f = global_id_0 * 10.0f + global_id_1 * 1.0f;
   f += local_id_0 * 0.1f + local_id_1 * 0.01f;

   output[index] = f;
}
*/