use ocl::builders::ProgramBuilder;
use ocl::core::{MemObjectType};
use ocl::{ Context, Device, DeviceType,  Platform, Queue, MemFlags, Image};
use ocl::enums::{ ImageChannelDataType, ImageChannelOrder,  };
use core::f32;
use png::{BitDepth, ColorType, Decoder, Encoder};
use std::{io::BufWriter};

const SCALE: usize = 10;
fn main() -> Result<(), Box<dyn std::error::Error>> {

    let platforms = Platform::list();

    let platform = platforms.into_iter().next().expect("No platforms found");

    let mut devices = Device::list(platform, Some(DeviceType::GPU)).unwrap_or_default();
    if devices.is_empty() {
        devices = Device::list(platform, Some(DeviceType::CPU)).unwrap_or_default();
    }
    let dev = devices.into_iter().next().expect("No devices found");

    let context = Context::builder().platform(platform).devices(dev).build()?;
    let queue = Queue::new(&context, (*dev).into(), None)?;
    let kernel_source = format!(
        r#"
constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE
   | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
   
__kernel void interp(read_only image2d_t src_image,
                     write_only image2d_t dst_image) {{
   float4 pixel;
   float2 input_coord = (float2)
      (get_global_id(0) + (1.0f/({SCALE}*2)),
       get_global_id(1) + (1.0f/({SCALE}*2)));
   int2 output_coord = (int2)
      ({SCALE}*get_global_id(0),
       {SCALE}*get_global_id(1));
   for(int i=0; i<{SCALE}; i++) {{
      for(int j=0; j<{SCALE}; j++) {{
         pixel = read_imagef(src_image, sampler,
           (float2)(input_coord + 
           (float2)(1.0f*i/{SCALE}, 1.0f*j/{SCALE})));
         write_imagef(dst_image, output_coord + 
                      (int2)(i, j), pixel);
      }} 
   }}
}}
"#,
    );

    let program_con = ProgramBuilder::new()
        .src(&kernel_source)
        .devices(dev.clone())
        .build(&context)?;

    let file = std::fs::File::open("input.png")?;
    let decoder = Decoder::new(file);
    let mut reader = decoder.read_info()?;
    
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    
    let width = info.width as usize;
    let height = info.height as usize;
    
    let grayscale_data: Vec<u16> = match info.color_type {
        ColorType::Grayscale => {
            match info.bit_depth {
                BitDepth::Eight => buf.into_iter().map(|x| (x as u16) << 8).collect(),
                BitDepth::Sixteen => {
                    let mut result = Vec::with_capacity(width * height);
                    for chunk in buf.chunks(2) {
                        let val = u16::from_be_bytes([chunk[0], chunk[1]]);
                        result.push(val);
                    }
                    result
                }
                _ => return Err("Unsupported bit depth".into()),
            }
        }
        ColorType::Rgb => {
            let mut result = Vec::with_capacity(width * height);
            for chunk in buf.chunks(3) {
                let gray = (0.299 * chunk[0] as f32 + 0.587 * chunk[1] as f32 + 0.114 * chunk[2] as f32) as u16;
                result.push(gray << 8);
            }
            result
        }
        ColorType::Rgba => {
            let mut result = Vec::with_capacity(width * height);
            for chunk in buf.chunks(4) {
                let gray = (0.299 * chunk[0] as f32 + 0.587 * chunk[1] as f32 + 0.114 * chunk[2] as f32) as u16;
                result.push(gray << 8);
            }
            result
        }
        _ => return Err("Unsupported color type".into()),
    };
    
    println!("Input image: {}x{}", width, height);
    
    let mut rgba_input = Vec::with_capacity(grayscale_data.len() * 4);
    for &pixel in &grayscale_data {
        rgba_input.push(pixel);
        rgba_input.push(pixel);
        rgba_input.push(pixel);
        rgba_input.push(65535);
    }

    let dst_width = width * SCALE;
    let dst_height = height * SCALE;


    let src_image = Image::<u16>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt16)
        .image_type(MemObjectType::Image2d)
        .flags(MemFlags::new().read_only() | MemFlags::new().copy_host_ptr())
        .copy_host_slice(&rgba_input)
        .dims([width, height])
        .queue(queue.clone())
        .build()?;
    let dst_image = Image::<u16>::builder()
        .channel_order(ImageChannelOrder::Rgba)
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

    unsafe {
        kernel.cmd()
            .queue(&queue)
            .global_work_size([width, height])
            .enq()?;
    }

    let mut output_data = vec![0u16; dst_width * dst_height * 4]; // 4 channels (RGBA)
    dst_image.read(&mut output_data).enq()?;
    
    let mut grayscale_output = Vec::with_capacity(dst_width * dst_height);
    for chunk in output_data.chunks(4) {
        grayscale_output.push(chunk[0]);
    }
    
    let file1 = std::fs::File::create("output.png")?;
    let ref mut w = BufWriter::new(file1);
    
    let mut encoder = Encoder::new(w, dst_width as u32, dst_height as u32);
    encoder.set_color(ColorType::Grayscale);
    encoder.set_depth(BitDepth::Sixteen);
    
    let mut writer = encoder.write_header()?;
    
    let mut byte_data = Vec::with_capacity(grayscale_output.len() * 2);
    for &pixel in &grayscale_output {
        byte_data.extend_from_slice(&pixel.to_be_bytes());
    }
    
    writer.write_image_data(&byte_data)?;
    writer.finish()?;
    println!("Output image written to output.png ({}x{})", dst_width, dst_height);


    Ok(())
}