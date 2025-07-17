use ocl::{Platform, Device, Context, Queue, Program, Buffer, flags, DeviceType, builders::KernelBuilder};
use std::time::Instant;

const ARRAY_SIZE: usize = 65536;
const NUM_KERNELS: usize = 2;

fn main() -> ocl::Result<()> {
    let src = std::fs::read_to_string("hello_kernel.cl").expect("Failed to read reduction.cl");

    let platform = Platform::list().into_iter().next().unwrap();
    let mut devices = Device::list(platform, Some(DeviceType::GPU)).unwrap_or_default();
    if devices.is_empty() {
        devices = Device::list(platform, Some(DeviceType::CPU)).unwrap_or_default();
    }
    let device = devices.into_iter().next().unwrap();

    let context = Context::builder().platform(platform).devices(device.clone()).build()?;
    let queue = Queue::new(&context, device.clone(), Some(flags::QUEUE_PROFILING_ENABLE))?;

    let group_size = match device.info(ocl::core::DeviceInfo::MaxWorkGroupSize)? {
    ocl::enums::DeviceInfoResult::MaxWorkGroupSize(size) => size,
    _ => 256,
    };
    let mut global_size = ARRAY_SIZE / 4;

    let mut data = vec![0.0f32; ARRAY_SIZE];
    for i in 0..ARRAY_SIZE {
        data[i] = i as f32;
    }

    let data_buffer = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .len(ARRAY_SIZE)
        .copy_host_slice(&data)
        .build()?;

    let program = Program::builder()
        .src(src)
        .devices(device.clone())
        .build(&context)?;

    let kernel_names = ["reduction_vector", "reduction_complete"];

    let start0 = Instant::now();

    let kernel_vec = KernelBuilder::new()
        .program(&program)
        .name(kernel_names[0])
        .queue(queue.clone())
        .arg(&data_buffer)
        .arg_local::<f32>(group_size)
        .global_work_size(global_size)
        .local_work_size(group_size)
        .build()?;

    unsafe {
        kernel_vec.cmd()
            .queue(&queue)
            .global_work_size(global_size)
            .local_work_size(group_size)
            .enq()?;
    }
    let duration0 = start0.elapsed();
    let start1 = Instant::now();
    global_size /= group_size;
    while global_size > group_size {
        unsafe {
            kernel_vec.cmd()
                .queue(&queue)
                .global_work_size(global_size)
                .local_work_size(group_size)
                .enq()?;
        }
        global_size /= group_size;
    }

    let sums_buffer = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_WRITE_ONLY)
        .len(global_size)
        .build()?;

    let kernel_com = KernelBuilder::new()
        .program(&program)
        .name(kernel_names[1])
        .queue(queue.clone())
        .arg(&data_buffer)
        .arg_local::<f32>(global_size*4)
        .arg(&sums_buffer)
        .global_work_size(global_size)
        .local_work_size(global_size)
        .build()?;


    unsafe {
        kernel_com.cmd()
            .queue(&queue)
            .global_work_size(global_size)
            .local_work_size(global_size)
            .enq()?;
    }

    queue.finish()?;

    let mut sums = vec![0.0f32; global_size];
    sums_buffer.read(&mut sums[..]).enq()?;
    let sum: f32 = sums.iter().sum();

    for i in 0..NUM_KERNELS {
        println!("{} sum is: {}", kernel_names[i], sum);
        let actual_sum = (ARRAY_SIZE as f32 / 2.0) * ((ARRAY_SIZE - 1) as f32);
        if (sum - actual_sum).abs() > 0.01 * sum.abs() {
            println!("Check failed.");
        } else {
            println!("Check passed.");
        }
    }
    let duration1 = start1.elapsed();
    
    println!("Time elapsed: {:.6} seconds", duration0.as_secs_f64());
    println!("Time elapsed: {} µs", duration0.as_micros());

    println!("Time elapsed: {:.6} seconds", duration1.as_secs_f64());
    println!("Time elapsed: {} µs", duration1.as_micros());
    Ok(())
}