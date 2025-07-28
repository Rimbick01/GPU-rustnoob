use ocl::{builders::KernelBuilder, flags, Buffer, Context, Device, DeviceType, Platform, Program, Queue};
use std::time::Instant;
use rand::Rng;

const DIRECTION: i32 = 0;
const NUM_FLOATS: usize = 1048576;

fn main() -> ocl::Result<()> {
    println!("Bitonic Sort - Processing {} floats", NUM_FLOATS);
    
    let src = std::fs::read_to_string("hello_kernel.cl").expect("Failed to read bsort.cl");
    
    let platform = Platform::list().into_iter().next().unwrap();
    let mut devices = Device::list(platform, Some(DeviceType::GPU)).unwrap_or_default();
    if devices.is_empty() {
        devices = Device::list(platform, Some(DeviceType::CPU)).unwrap_or_default();
    }
    let device = devices.into_iter().next().unwrap();
    
    let context = Context::builder()
        .platform(platform)
        .devices(device.clone())
        .build()?;
    let queue = Queue::new(&context, device.clone(), None)?;
    
    println!("Generating random data...");
    let start_time = Instant::now();
    let mut rng = rand::thread_rng();
    let mut data: Vec<f32> = (0..NUM_FLOATS)
        .map(|_| rng.r#gen::<f32>() * 1000000.0)
        .collect();
    println!("Data generation took: {:?}", start_time.elapsed());
    
    println!("First 16 values: {:?}", &data[0..16]);
    
    let program = Program::builder()
        .src(src)
        .devices(device.clone())
        .build(&context)?;
    
    let max_wg_size = device.max_wg_size()?;
    let mut local_size = 1;
    while local_size * 2 <= max_wg_size {
        local_size *= 2;
    }
    
    let global_size = NUM_FLOATS / 8;
    if global_size < local_size {
        local_size = global_size;
    }
    
    println!("Local work size: {}", local_size);
    println!("Global work size: {}", global_size);
    println!("Number of work groups: {}", global_size / local_size);
    
    let data_buffer = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
        .len(NUM_FLOATS)
        .copy_host_slice(&data)
        .build()?;
    
    let local_memory_size = 8 * local_size;
    
    println!("Starting bitonic sort...");
    let sort_start = Instant::now();
    
    let kernel_init = KernelBuilder::new()
        .program(&program)
        .name("bsort_init")
        .queue(queue.clone())
        .arg(&data_buffer)
        .arg_local::<f32>(local_memory_size)
        .global_work_size(global_size)
        .local_work_size(local_size)
        .build()?;
    
    println!("Executing initial sort...");
    unsafe {
        kernel_init.cmd()
            .queue(&queue)
            .enq()?;
    }
    
    let num_stages = global_size / local_size;
    println!("Number of stages: {}", num_stages);
    
    let mut high_stage = 2;
    while high_stage < num_stages {
        println!("Processing high_stage: {}", high_stage);
        
        let mut stage = high_stage;
        while stage > 1 {
            let kernel_stage_n = KernelBuilder::new()
                .program(&program)
                .name("bsort_stage_n")
                .queue(queue.clone())
                .arg(&data_buffer)
                .arg_local::<f32>(local_memory_size)
                .arg(stage as i32)
                .arg(high_stage as i32)
                .global_work_size(global_size)
                .local_work_size(local_size)
                .build()?;
                
            unsafe {
                kernel_stage_n.cmd()
                    .queue(&queue)
                    .enq()?;
            }
            
            stage >>= 1;
        }
        
        let kernel_stage_0 = KernelBuilder::new()
            .program(&program)
            .name("bsort_stage_0")
            .queue(queue.clone())
            .arg(&data_buffer)
            .arg_local::<f32>(local_memory_size)
            .arg(high_stage as i32)
            .global_work_size(global_size)
            .local_work_size(local_size)
            .build()?;
            
        unsafe {
            kernel_stage_0.cmd()
                .queue(&queue)
                .enq()?;
        }
        
        high_stage <<= 1;
    }
    
    println!("Performing bitonic merge...");
    let mut stage = num_stages;
    while stage > 1 {
        let kernel_merge = KernelBuilder::new()
            .program(&program)
            .name("bsort_merge")
            .queue(queue.clone())
            .arg(&data_buffer)
            .arg_local::<f32>(local_memory_size)
            .arg(stage as i32)
            .arg(DIRECTION)
            .global_work_size(global_size)
            .local_work_size(local_size)
            .build()?;
            
        unsafe {
            kernel_merge.cmd()
                .queue(&queue)
                .enq()?;
        }
        
        stage >>= 1;
    }
    
    let kernel_merge_last = KernelBuilder::new()
        .program(&program)
        .name("bsort_merge_last")
        .queue(queue.clone())
        .arg(&data_buffer)
        .arg_local::<f32>(local_memory_size)
        .arg(DIRECTION)
        .global_work_size(global_size)
        .local_work_size(local_size)
        .build()?;
        
    unsafe {
        kernel_merge_last.cmd()
            .queue(&queue)
            .enq()?;
    }
    
    println!("Sorting completed in: {:?}", sort_start.elapsed());
    
    println!("Reading results...");
    let read_start = Instant::now();
    data_buffer.read(&mut data).enq()?;
    println!("Data read took: {:?}", read_start.elapsed());
    
    println!("First 16 sorted values: {:?}", &data[0..16]);
    println!("Last 16 sorted values: {:?}", &data[NUM_FLOATS-16..NUM_FLOATS]);
    
    println!("Verifying sort...");
    let verify_start = Instant::now();
    let mut check = true;
    
    if DIRECTION == 0 {
        for i in 1..NUM_FLOATS {
            if data[i] < data[i-1] {
                check = false;
                println!("Sort failed at index {}: {} > {}", i, data[i-1], data[i]);
                break;
            }
        }
    } else {
        for i in 1..NUM_FLOATS {
            if data[i] > data[i-1] {
                check = false;
                println!("Sort failed at index {}: {} < {}", i, data[i-1], data[i]);
                break;
            }
        }
    }
    
    println!("Verification took: {:?}", verify_start.elapsed());
    
    println!("\n=== RESULTS ===");
    println!("Local size: {}", local_size);
    println!("Global size: {}", global_size);
    println!("Direction: {}", if DIRECTION == 0 { "Ascending" } else { "Descending" });
    println!("Data size: {} floats ({:.2} MB)", NUM_FLOATS, NUM_FLOATS as f32 * 4.0 / 1024.0 / 1024.0);
    
    if check {
        println!("Bitonic sort SUCCEEDED!");
    } else {
        println!("Bitonic sort FAILED!");
    }
    
    Ok(())
}
