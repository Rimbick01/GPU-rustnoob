use ocl::{builders::ProgramBuilder, core::{build_program, MemObjectType}, enums::{ImageChannelDataType, ImageChannelOrder}, Context, Device, DeviceType, Image, MemFlags, Platform, Queue};
use std::ffi::CString;
use image::{ GrayImage, Luma};

const SCALE_FACTOR: usize = 10;
const INPUT_FILE: &str = "input.png";
const OUTPUT_FILE: &str = "output.png";
const KERNEL_FILE: &str = "hello_kernel.cl";

fn main() -> ocl::Result<()> {
    let img = image::open(INPUT_FILE).expect("Failed to open input image").to_luma16();
    let (width, height) = img.dimensions();
    let dst_width = width * SCALE_FACTOR as u32;
    let dst_height = height * SCALE_FACTOR as u32;

    let src_data: Vec<u16> = img.pixels().flat_map(|p| p.0).collect();

    let platform = Platform::list().into_iter().next().unwrap();
    let mut devices = Device::list(platform, Some(DeviceType::GPU)).unwrap_or_default();
    if devices.is_empty() {
        devices = Device::list(platform, Some(DeviceType::CPU)).unwrap_or_default();
    }
    let dev = devices.into_iter().next().unwrap();
    let context = Context::builder().platform(platform).devices(dev).build()?;
    let queue = Queue::new(&context, dev.clone(), None)?;

    let program_handle = std::fs::read_to_string(KERNEL_FILE).unwrap();

    let program_con = ProgramBuilder::new()
        .src(&program_handle)
        .devices(dev.clone())
        .build(&context)?;
    let build_opts = CString::new(format!("-DSCALE={}", SCALE_FACTOR)).unwrap();
    build_program::<()>(&program_con, None, &build_opts, None, None)?;

    let src_image = Image::<u16>::builder()
        .channel_order(ImageChannelOrder::Luminance)
        .channel_data_type(ImageChannelDataType::UnormInt16)
        .image_type(MemObjectType::Image2d)
        .flags(MemFlags::new().read_only() | MemFlags::new().copy_host_ptr())
        .copy_host_slice(&src_data)
        .dims([width, height])
        .queue(queue.clone())
        .build()?;
    let dst_image = Image::<u16>::builder()
        .channel_order(ImageChannelOrder::Luminance)
        .channel_data_type(ImageChannelDataType::UnormInt16)
        .image_type(MemObjectType::Image2d)
        .flags(MemFlags::new().write_only())
        .dims([dst_width, dst_height])
        .queue(queue.clone())
        .build()?;

    let kernel = ocl::Kernel::builder()
        .program(&program_con)
        .name("interp")
        .queue(queue.clone())
        .arg(&src_image)
        .arg(&dst_image)
        .global_work_size([width, height])
        .build()?;

    unsafe {kernel.cmd().queue(&queue) .global_work_size([width, height]) .enq()?; }

    let mut dst_data = vec![0u16; (dst_width * dst_height) as usize];
    dst_image.read(&mut dst_data).enq()?;

    let mut out_img = GrayImage::new(dst_width, dst_height);
    for y in 0..dst_height {
        for x in 0..dst_width {
            let idx = (y * dst_width + x) as usize;
            out_img.put_pixel(x, y, Luma([dst_data[idx].try_into().unwrap()]));
        }
    }
    out_img.save(OUTPUT_FILE).expect("Failed to save output image");

    println!("Wrote upscaled image to {}", OUTPUT_FILE);
    Ok(())
}