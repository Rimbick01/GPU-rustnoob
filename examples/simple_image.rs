use ocl::builders::ProgramBuilder;
use ocl::core::MemObjectType;
use ocl::{ Context, Device, DeviceType,  Platform, Queue, MemFlags, Image};
use ocl::enums::{ ImageChannelDataType, ImageChannelOrder,  };
use std::{fs};

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

    let program_con = ProgramBuilder::new()
        .src(&program_handle)
        .devices(dev.clone())
        .build(&context)?;
    let width = 4;
    let height = 4;

    let mut src_data = vec![0u32; width * height * 4];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 4;
            src_data[idx + 0] = (x + y * width) as u32; // .x channel
            src_data[idx + 1] = 0;
            src_data[idx + 2] = 0;
            src_data[idx + 3] = 255;
        }
    }

    let src_image = Image::<u32>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnsignedInt32)
        .image_type(MemObjectType::Image2d)
        .flags(MemFlags::new().read_only() | MemFlags::new().copy_host_ptr())
        .copy_host_slice(&src_data)
        .dims([width, height])
        .queue(queue.clone())
        .build()?;
    let dst_image = Image::<u32>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnsignedInt32)
        .image_type(MemObjectType::Image2d)
        .flags(MemFlags::new().write_only())
        .copy_host_slice(&src_data)
        .dims([width, height])
        .queue(queue.clone())
        .build()?;

    let kernel = ocl::Kernel::builder()
        .program(&program_con)
        .name("simple_image")
        .queue(queue.clone())
        .arg(&src_image)
        .arg(&dst_image)
        .global_work_size([width, height])
        .build()?;

    unsafe {
        kernel.cmd()
            .queue(&queue)
            .global_work_size([width, height])
            .enq()?;
    }

    let mut dst_data = vec![0u32; width * height * 4];
    dst_image.read(&mut dst_data).enq()?;

    println!("Output image .x channel:");
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 4;
            print!("{:10} ", dst_data[idx]);
        }
        println!();
    }

    Ok(())
}
/*
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
      CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST; 

__kernel void simple_image(read_only image2d_t src_image,
                        write_only image2d_t dst_image) {

   uint offset = get_global_id(1) * 0x4000 + get_global_id(0) * 0x1000;

   int2 coord = (int2)(get_global_id(0), get_global_id(1));
   uint4 pixel = read_imageui(src_image, sampler, coord);

   pixel.x -= offset;

   write_imageui(dst_image, coord, pixel);
}
 */