use ocl::{Device, enums::DeviceInfo};

fn main() {
    let platforms = ocl::Platform::list();
    println!("Number of platforms: {}", platforms.len());

    for (p_idx,platform) in platforms.iter().enumerate() {
        println!("Platform{}:", p_idx);
        println!("CL_PLATFORM_NAME:{}", platform.name().unwrap());
        println!("CL_PLATFORM_VENDOR:{}", platform.vendor().unwrap());
        println!("CL_PLATFORM_VERSION:{}", platform.version().unwrap());
        println!("CL_PLATFORM_PROFILE:{}", platform.profile().unwrap());
        println!("CL_PLATFORM_EXTENSIONS:{:?}", platform.extensions().unwrap());

        let devices = Device::list_all(platform).unwrap();
        println!("  Number of devices: {}", devices.len());

        for (d_idx,device) in devices.iter().enumerate(){
            println!("\tDevice {}:", d_idx);
            println!("\tCL_DEVICE_NAME:    {}", device.name().unwrap());
            println!("\tCL_DEVICE_VENDOR:  {}", device.vendor().unwrap());
            println!("\tCL_DEVICE_VERSION: {}", device.version().unwrap());
            println!("\tCL_DEVICE_PROFILE: {}", device.info(DeviceInfo::Profile).unwrap());
            println!("\tCL_DEVICE_EXTENSIONS: {}", device.info(DeviceInfo::Extensions).unwrap());
            println!("\tCL_DEVICE_VERSION:    {:?}", device.info(DeviceInfo::Type).unwrap());
            println!("\tCL_DEVICE_VENDOR_ID: {:?}", device.info(DeviceInfo::VendorId).unwrap());
            println!("\tCL_DEVICE_OPENCL_C_VERSION: {:?}", device.info(DeviceInfo::OpenclCVersion).unwrap());
            println!("\tCL_DEVICE_BUILT_IN_KERNELS: {:?}", device.info(DeviceInfo::BuiltInKernels).unwrap());
            // println!("\tCL_DEVICE_SVM_CAPABILITIES: {:?}", device.info(DeviceInfo::SvmCapabilities).unwrap());
            println!();
        }
        println!();
    }
}