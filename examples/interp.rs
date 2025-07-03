use ocl::{builders::ProgramBuilder, core::build_program, enums::{ImageChannelDataType, ImageChannelOrder}, Context, Device, DeviceType, Image, MemFlags, Platform, Queue};
use std::ffi::CString;
use image::{ GenericImageView, RgbaImage};

const INPUT_FILE: &str = "input.png";
const OUTPUT_FILE: &str = "output.png";
const KERNEL_FILE: &str = "hello_kernel.cl";

fn main() -> ocl::Result<()> {
    let img = image::open(INPUT_FILE).expect("Failed to open input image");
    let (src_width, src_height) = img.dimensions();
    let scale = 2;
    let dst_width = src_width * scale;
    let dst_height = src_height * scale;

    let img_rgba: RgbaImage = img.to_rgba8();
    let mut src_data = Vec::with_capacity((src_width * src_height * 4) as usize);
    for p in img_rgba.pixels() {
        src_data.push(p[0] as f32 / 255.0);
        src_data.push(p[1] as f32 / 255.0);
        src_data.push(p[2] as f32 / 255.0);
        src_data.push(p[3] as f32 / 255.0);
    }

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
    let build_opts = CString::new(format!("-DSCALE={}", scale)).unwrap();
    build_program::<()>(&program_con, None, &build_opts, None, None)?;

    let src_image = Image::<f32>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::Float)
        .flags(MemFlags::new().read_only() | MemFlags::new().copy_host_ptr())
        .copy_host_slice(&src_data)
        .dims([src_width, src_height])
        .queue(queue.clone())
        .build()?;
    let dst_image = Image::<f32>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::Float)
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
        .global_work_size([src_width, src_height])
        .build()?;

    unsafe {
        kernel.cmd()
            .queue(&queue)
            .global_work_size([src_width, src_height])
            .enq()?;
    }

    let mut dst_data = vec![0.0f32; (dst_width * dst_height * 4) as usize];
    dst_image.read(&mut dst_data).enq()?;

    let mut out_img = RgbaImage::new(dst_width, dst_height);
    for y in 0..dst_height {
        for x in 0..dst_width {
            let idx = ((y * dst_width + x) * 4) as usize;
            let r = (dst_data[idx + 0].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (dst_data[idx + 1].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (dst_data[idx + 2].clamp(0.0, 1.0) * 255.0) as u8;
            let a = (dst_data[idx + 3].clamp(0.0, 1.0) * 255.0) as u8;
            out_img.put_pixel(x, y, image::Rgba([r, g, b, a]));
        }
    }
    out_img.save(OUTPUT_FILE).expect("Failed to save output image");

    println!("Wrote upscaled image to {}", OUTPUT_FILE);
    Ok(())
}
