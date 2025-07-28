use ocl::{builders::KernelBuilder, flags, Buffer, Context, Device, DeviceType, Platform, Program, Queue};

const ASCENDING: i32 = 0;
const DESCENDING: i32 = -1;

fn main() -> ocl::Result<()> {
    let src = std::fs::read_to_string("hello_kernel.cl").expect("Failed to read bsort8.cl");
    
    let platform = Platform::list().into_iter().next().unwrap();
    let mut devices = Device::list(platform, Some(DeviceType::GPU)).unwrap_or_default();
    if devices.is_empty() {
        devices = Device::list(platform, Some(DeviceType::CPU)).unwrap_or_default();
    }
    let device = devices.into_iter().next().unwrap();
    
    let context = Context::builder() .platform(platform).devices(device.clone()) .build()?;
    let queue = Queue::new(&context, device.clone(), None)?;
    
    let mut data: [f32; 8] = [3.0, 5.0, 4.0, 6.0, 0.0, 7.0, 2.0, 1.0];
    println!("Input:  {:3.1} {:3.1} {:3.1} {:3.1} {:3.1} {:3.1} {:3.1} {:3.1}",
             data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);
    
    let program = Program::builder()
        .src(src)
        .devices(device.clone())
        .build(&context)?;
    
    let data_buffer = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
        .len(8)
        .copy_host_slice(&data)
        .build()?;
    
    let kernel = KernelBuilder::new()
        .program(&program)
        .name("bsort8")
        .queue(queue.clone())
        .arg(&data_buffer)
        .arg(ASCENDING)
        .build()?;
    
    unsafe {
        kernel.cmd()
            .queue(&queue)
            .global_work_size(1)
            .enq()?;
    }
    
    data_buffer.read(&mut data[..]).enq()?;
    
    println!("Output: {:3.1} {:3.1} {:3.1} {:3.1} {:3.1} {:3.1} {:3.1} {:3.1}",
             data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);
    
    let mut check = true;
    if ASCENDING == 0 {
        for i in 1..8 {
            if data[i] < data[i - 1] {
                check = false;
                break;
            }
        }
    } else if ASCENDING == DESCENDING {
        for i in 1..8 {
            if data[i] > data[i - 1] {
                check = false;
                break;
            }
        }
    }
    
    if check {
        println!("Bitonic sort succeeded.");
    } else {
        println!("Bitonic sort failed.");
    }
    
    Ok(())
}