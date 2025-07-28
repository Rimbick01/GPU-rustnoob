use ocl::{Platform, Device, Context, Queue, Program, Buffer, flags, DeviceType, builders::KernelBuilder};
use ocl::prm::Char16;
const TEXT_FILE: &str = "kafka.txt";

fn main() -> ocl::Result<()> {
    let src = std::fs::read_to_string("hello_kernel.cl").expect("Failed to read ");
    let text_file = std::fs::read_to_string(TEXT_FILE).expect("Failed to read ");
    let text_size = text_file.len();

    let platform = Platform::list().into_iter().next().unwrap();
    let mut devices = Device::list(platform, Some(DeviceType::GPU)).unwrap_or_default();
    if devices.is_empty() {
        devices = Device::list(platform, Some(DeviceType::CPU)).unwrap_or_default();
    }
    let device = devices.into_iter().next().unwrap();

    let context = Context::builder().platform(platform).devices(device.clone()).build()?;
    let queue = Queue::new(&context, device.clone(), Some(flags::QUEUE_PROFILING_ENABLE))?;

    let mut result = [0i32; 4];
    let max_group_size = match device.info(ocl::core::DeviceInfo::MaxWorkGroupSize)? {
    ocl::enums::DeviceInfoResult::MaxWorkGroupSize(size) => size,
    _ => 0,
    };
    let max_compute_units = match device.info(ocl::core::DeviceInfo::MaxComputeUnits)? {
    ocl::enums::DeviceInfoResult::MaxComputeUnits(units) => units,
    _ => 0,
    };
    let global_size = (max_group_size as usize) * (max_compute_units as usize) ;
    let local_size = max_group_size as usize;
    
    let chars_per_item = (text_size as usize / global_size as usize) + 1;
    
    let pattern = "thatwithhavefrom";
    let mut pattern_array = [0i8; 16];
    for (i, &b) in pattern.as_bytes().iter().enumerate() {
        pattern_array[i] = b as i8;
    }

    let pattern_vec = Char16::new(
    pattern_array[0], pattern_array[1], pattern_array[2], pattern_array[3],
    pattern_array[4], pattern_array[5], pattern_array[6], pattern_array[7],
    pattern_array[8], pattern_array[9], pattern_array[10], pattern_array[11],
    pattern_array[12], pattern_array[13], pattern_array[14], pattern_array[15],
    );

    let program = Program::builder()
        .src(src)
        .devices(device.clone())
        .build(&context)?;

    let text_buffer = Buffer::<u8>::builder()
        .queue(queue.clone())
        .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_COPY_HOST_PTR)
        .len(text_size)
        .copy_host_slice(text_file.as_bytes())
        .build()?;
    
    let result_buffer = Buffer::<i32>::builder()
        .queue(queue.clone())
        .flags(ocl::flags::MEM_READ_WRITE | ocl::flags::MEM_COPY_HOST_PTR)
        .len(4)
        .copy_host_slice(&result)
        .build()?;
    

    let kernel_vec = KernelBuilder::new()
        .program(&program)
        .name("string_search")
        .queue(queue.clone())
        .arg(pattern_vec)    
        .arg(&text_buffer)
        .arg(chars_per_item as i32)
        .arg_local::<i32>(4)
        .arg(&result_buffer)
        .global_work_size(global_size)
        .local_work_size(local_size)
        .build()?;

    unsafe { kernel_vec.cmd() .queue(&queue) .global_work_size(global_size).local_work_size(local_size) .enq()?; }
    result_buffer.cmd().offset(0).read(&mut result[..]).enq()?;
    
    println!("\nResults: ");
    println!("Number of occurrences of 'that': {}", result[0]);
    println!("Number of occurrences of 'with': {}", result[1]);
    println!("Number of occurrences of 'have': {}", result[2]);
    println!("Number of occurrences of 'from': {}", result[3]);
    
    Ok(())
}