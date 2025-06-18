use ocl::{ flags, Context, Device, DeviceType, Platform};

fn main() -> ocl::Result<()> {

    let platforms = Platform::list();

    let platform = platforms.into_iter().next().expect("No platforms found");

    let mut devices = Device::list(platform, Some(DeviceType::GPU)).unwrap_or_default();
    if devices.is_empty() {
        devices = Device::list(platform, Some(DeviceType::CPU)).unwrap_or_default();
    }
    let dev = devices.into_iter().next().expect("No devices found");

    let context = Context::builder().platform(platform).devices(dev).build()?;
    let main_data = vec![0.0f32; 100];

    let main_buffer = ocl::Buffer::<f32>::builder().context(&context).len(main_data.len()).copy_host_slice(&main_data).flags(flags::MEM_READ_ONLY).build()?;
    // let sub_buffer = ocl::Buffer::<f32>::builder().parent(&main_buffer).region(MemoryRegion::new(30 * std::mem::size_of::<f32>(), 20 * std::mem::size_of::<f32>())).flags(flags::MEM_READ_ONLY).build()?;
    // let sub_buffer = main_buffer.create_sub_buffer(Some(flags::MEM_READ_ONLY),BufferCreateType::Region,BufferRegion::new(30 * std::mem::size_of::<f32>(), 20 * std::mem::size_of::<f32>())).unwrap();
    let offset_elements = 30; // Offset in elements
    let size_elements = 20;   // Size in elements
    // let sub_region = BufferRegion::new(
    //     offset_elements * std::mem::size_of::<f32>(), // Offset in bytes
    //     size_elements * std::mem::size_of::<f32>(),    // Size in bytes
    // );
    let sub_buffer = main_buffer.create_sub_buffer(
        Some(flags::MEM_READ_ONLY),
    //     BufferCreateType::Region,
    //     sub_region,
        offset_elements,
        size_elements
    )?;
    println!("Main buffer size: {} bytes", main_buffer.len());
    println!("Main buffer size: {} bytes", sub_buffer.len());

    println!("Main array address: {:p}", main_data.as_ptr());
 
    Ok(())
}