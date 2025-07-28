use ocl::{builders::KernelBuilder, flags, Buffer, Context, Device, DeviceType, Platform, Program, Queue};
use rand::Rng;

const DIRECTION: i32 = 0;
const NUM_FLOATS: usize = 8192*4;

fn main() -> ocl::Result<()> {
    let src = std::fs::read_to_string("hello_kernel.cl").expect("Failed to read bsort.cl");
    
    let platform = Platform::list().into_iter().next().unwrap();
    let mut devices = Device::list(platform, Some(DeviceType::GPU)).unwrap_or_default();
    if devices.is_empty() {
        devices = Device::list(platform, Some(DeviceType::CPU)).unwrap_or_default();
    }
    let device = devices.into_iter().next().unwrap();
    
    let context = Context::builder() .platform(platform).devices(device.clone()) .build()?;
    let queue = Queue::new(&context, device.clone(), None)?;
    
    let mut data: Vec<f32> = (0..NUM_FLOATS)
        .map(|_| rand::thread_rng().r#gen::<f32>() * 10000.0)
        .collect();
    
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
    
    let data_buffer = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
        .len(NUM_FLOATS)
        .copy_host_slice(&data)
        .build()?;
    
    let local_memory_size = 8 * local_size;
    
    let kernel = KernelBuilder::new()
        .program(&program)
        .name("bsort_init")
        .queue(queue.clone())
        .arg(&data_buffer)
        .arg_local::<f32>(local_memory_size)
        .global_work_size(global_size)
        .local_work_size(local_size)
        .build()?;
    
    unsafe {
        kernel.cmd()
            .queue(&queue)
            .global_work_size(global_size)
            .local_work_size(local_size)
            .enq()?;
    }
    
    data_buffer.read(&mut data).enq()?;
    
    println!("First 16 sorted values: {:?}", &data[0..16]);
    println!("Last 16 sorted values: {:?}", &data[NUM_FLOATS-16..NUM_FLOATS]);
    
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
    println!("Data size: {} floats ({:.2} MB)", NUM_FLOATS, NUM_FLOATS as f32 * 4.0 / 1024.0 / 1024.0);
    
    if check {
        println!("Bitonic sort SUCCEEDED!");
    } else {
        println!("Bitonic sort FAILED!");
    }
    Ok(())
}