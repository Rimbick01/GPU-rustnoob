use ocl::{Device, enums::DeviceInfo};

fn main() {
    let platforms = ocl::Platform::list();
    println!("Number of platforms: {}", platforms.len());

    for platform in platforms.iter() {
        println!("Platform {}:", platform);
        println!("  Name:    {}", platform.name().unwrap());
        println!("  Vendor:  {}", platform.vendor().unwrap());
        println!("  Version: {}", platform.version().unwrap());
        println!("  Profile: {}", platform.profile().unwrap());
        println!("  Extensions: {:?}", platform.extensions().unwrap());

        let devices = Device::list_all(platform).unwrap();
        println!("  Number of devices: {}", devices.len());

        for device in devices.iter() {
            println!("    Device {}:", device);
            println!("      Name:    {}", device.name().unwrap());
            println!("      Vendor:  {}", device.vendor().unwrap());
            println!("      Version: {}", device.version().unwrap());
            println!("      Profile: {}", device.info(DeviceInfo::Profile).unwrap());
            println!("      Extensions: {}", device.info(DeviceInfo::Extensions).unwrap());
            println!("      Type:    {:?}", device.info(DeviceInfo::Type).unwrap());
            println!("      Vendor ID: {:?}", device.info(DeviceInfo::VendorId).unwrap());
            println!("      OpenCL C Version: {:?}", device.info(DeviceInfo::OpenclCVersion).unwrap());
            println!("      Built-in Kernels: {:?}", device.info(DeviceInfo::BuiltInKernels).unwrap());
            // println!("      SVM Capabilities: {:?}", device.info(DeviceInfo::SvmCapabilities).unwrap());
            println!();
        }
        println!();
    }
}