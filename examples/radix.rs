use ocl::{builders::KernelBuilder, flags, Buffer, Context, Device, DeviceType, Platform, Program, Queue};
use rand::Rng;


fn main() -> ocl::Result<()> {
    let src = std::fs::read_to_string("hello_kernel.cl").expect("Failed to read ");

    let platform = Platform::list().into_iter().next().unwrap();
    let mut devices = Device::list(platform, Some(DeviceType::GPU)).unwrap_or_default();
    if devices.is_empty() {
        devices = Device::list(platform, Some(DeviceType::CPU)).unwrap_or_default();
    }
    let device = devices.into_iter().next().unwrap();

    let context = Context::builder().platform(platform).devices(device.clone()).build()?;
    let queue = Queue::new(&context, device.clone(), Some(flags::QUEUE_PROFILING_ENABLE))?;
    let mut data = [0u16;8];
    for i in 0..8 {
       data[i] = i as u16;
    }
    for i in 0..7 {
      let j = i + rand::thread_rng().gen_range(0..(7-i));
    //   data[i as usize] = data[i as usize] ^ data[j]; data[j] =  data[i as usize] ^ data[j]; data[i as usize] = data[i as usize] ^ data[j];
        data.swap(i, j);
   }

   println!("Input: \n");
    for i in 0..8 {
      println!("data[]: {:.2}", data[i]);
   }

    let program = Program::builder()
        .src(src)
        .devices(device.clone())
        .build(&context)?;
    
    let data_buffer = Buffer::<u16>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
        .len(8)
        .copy_host_slice(&data)
        .build()?;
    
    let kernel = KernelBuilder::new()
        .program(&program)
        .name("radix_sort8")
        .queue(queue.clone())
        .arg(&data_buffer)
        .build()?;
    
    unsafe {
        kernel.cmd()
            .queue(&queue)
            .global_work_size(1)
            .enq()?;
    }
    
    data_buffer.read(&mut data[..]).enq()?;

    println!("Output: \n");
    for i in 0..8 {
      println!("data[]: {:.2}", data[i]);
   }

   let mut check = true;
    for i in 0..8 {
      if data[i] != i.try_into().unwrap() {
         check = false;
         break;
      }
   }
   if check {
      println!("The radix sort succeeded.\n");
   }else {
      println!("The radix sort failed.\n");
   }

    Ok(())
}