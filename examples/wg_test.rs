use ocl::builders::ProgramBuilder;
use ocl::core::DeviceInfo;
use ocl::{Buffer, Context, Device, DeviceType, MemFlags, Platform, Queue, core};
use std::fs;


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
    let program_con = ProgramBuilder::new().src(&program_handle).devices(dev.clone()).build(&context).unwrap(); 
    let a = Buffer::<u8>::builder().queue(queue.clone()).flags(MemFlags::new().write_only()).len(1).build()?;
    let b = Buffer::<u8>::builder().queue(queue.clone()).flags(MemFlags::new().write_only()).len(1).build()?;
    let kernel = ocl::Kernel::builder().program(&program_con).name("blank").queue(queue.clone()).arg(&a).arg(&b).build()?;

    unsafe {kernel.cmd().queue(&queue).global_work_size(1).enq()?;}
    println!("KERNELS work size: {:?}", dev.max_wg_size().unwrap());
    println!("KERNELS work size: {:?}", dev.max_wg_size().unwrap());

    let wg_size = match core::get_kernel_work_group_info(
        &kernel, dev.clone(), core::KernelWorkGroupInfo::WorkGroupSize
    )? {
        core::KernelWorkGroupInfoResult::WorkGroupSize(size) => size,
        _ => 0,
    };

    let wg_multiple = match core::get_kernel_work_group_info(
        &kernel, dev.clone(), core::KernelWorkGroupInfo::PreferredWorkGroupSizeMultiple
    )? {
        core::KernelWorkGroupInfoResult::PreferredWorkGroupSizeMultiple(mult) => mult,
        _ => 0,
    };

    let local_usage = match core::get_kernel_work_group_info(
        &kernel, dev.clone(), core::KernelWorkGroupInfo::LocalMemSize
    )? {
        core::KernelWorkGroupInfoResult::LocalMemSize(size) => size,
        _ => 0,
    };

    let private_usage = match core::get_kernel_work_group_info(
        &kernel, dev.clone(), core::KernelWorkGroupInfo::PrivateMemSize
    )? {
        core::KernelWorkGroupInfoResult::PrivateMemSize(size) => size,
        _ => 0,
    };
    let device_name = dev.name()?;
    let local_mem = dev.info(DeviceInfo::LocalMemSize)?;

    println!(
        "For the blank kernel running on the {} device, the maximum work-group size is {} and the work-group multiple is {}.\n",
         device_name, wg_size, wg_multiple
    );
    println!(
        "The kernel uses {} bytes of local memory out of a maximum of {} bytes. It uses {} bytes of private memory.",
        local_usage, local_mem, private_usage
    );

    Ok(())
}