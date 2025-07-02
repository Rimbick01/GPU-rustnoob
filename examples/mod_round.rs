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

    let program_con = ProgramBuilder::new().src(&program_handle) .devices(dev.clone()) .build(&context)?;

    let mod_input = [317.0f32, 23.0f32];
    let round_input = [-6.5f32, -3.5f32, 3.5f32, 6.5f32];

    let mut mod_output = [0.0f32; 2];
    let mut round_output = [[0.0f32; 4]; 5]; 

    let mod_input_buf = Buffer::<f32>::builder().queue(queue.clone()) .flags(flags::MEM_READ_ONLY |flags::MEM_COPY_HOST_PTR) .len(2) .copy_host_slice(&mod_input).build()?;
    let mod_output_buf = Buffer::<f32>::builder() .queue(queue.clone()).flags(flags::MEM_WRITE_ONLY) .len(2) .build()?;
    let round_input_buf = Buffer::<f32>::builder().queue(queue.clone()) .flags(flags::MEM_READ_ONLY |flags::MEM_COPY_HOST_PTR) .len(4) .copy_host_slice(&round_input).build()?;
    let round_output_buf = Buffer::<f32>::builder().queue(queue.clone()).flags(flags::MEM_WRITE_ONLY).len(5 * 4).build()?;

    let kernel = ocl::Kernel::builder()
        .program(&program_con)
        .name("mod_round")
        .queue(queue.clone())
        .arg(&mod_input_buf)
        .arg(&mod_output_buf)
        .arg(&round_input_buf)
        .arg(&round_output_buf)
        .global_work_size(1)
        .build()?;

    unsafe {kernel.cmd().queue(&queue) .global_work_size(1) .enq()?; }

    mod_output_buf.read(&mut mod_output[..]).enq()?;
    let mut round_output_flat = [0.0f32; 20];
    round_output_buf.read(&mut round_output_flat[..]).enq()?;
    for i in 0..5 {
        for j in 0..4 {
            round_output[i][j] = round_output_flat[i * 4 + j];
        }
    }

    println!("fmod({}, {}) = {}", mod_input[0], mod_input[1], mod_output[0]);
    println!("remainder({}, {}) = {}", mod_input[0], mod_input[1], mod_output[1]);
    println!("rounding input = ({:?})", round_input);
    let names = ["rint", "round", "ceil", "floor", "trunc"];
    for i in 0..5 {
        println!("{} = {:?}", names[i], round_output[i]);
    }

    println!();

    Ok(())
}
/*
__kernel void mod_round(__global float *mod_input, 
                        __global float *mod_output, 
                        __global float4 *round_input,
                        __global float4 *round_output) {

   mod_output[0] = fmod(mod_input[0], mod_input[1]);
   mod_output[1] = remainder(mod_input[0], mod_input[1]);
   
   round_output[0] = rint(*round_input);      
   round_output[1] = round(*round_input);
   round_output[2] = ceil(*round_input);
   round_output[3] = floor(*round_input);
   round_output[4] = trunc(*round_input);   
}
*/