use ocl::{Platform, Device, Context, Queue, Buffer, flags, DeviceType};

fn main() -> ocl::Result<()> {
    let platform = Platform::list().into_iter().next().unwrap();
    let mut devices = Device::list(platform, Some(DeviceType::GPU)).unwrap_or_default();
    if devices.is_empty() {
        devices = Device::list(platform, Some(DeviceType::CPU)).unwrap_or_default();
    }
    let device = devices.into_iter().next().unwrap();

    let context = Context::builder().platform(platform).devices(device.clone()).build()?;
    let queue = Queue::new(&context, device.clone(), None)?;

    let mut full_matrix = [0.0f32; 80];
    for i in 0..80 {
        full_matrix[i] = i as f32;
    }

    let matrix_buffer = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
        .len(full_matrix.len())
        .copy_host_slice(&full_matrix)
        .build()?;

    // Read the whole buffer linearly
    let mut read_back = [0.0f32; 80];
    matrix_buffer.read(&mut read_back[..]).enq()?;

    // Now extract the rectangle in Rust
    let mut zero_matrix = [0.0f32; 80];
    let src_cols = 10;
    let dst_cols = 10;
    let rect_cols = 4;
    let rect_rows = 4;
    let src_row0 = 3;
    let src_col0 = 5;
    let dst_row0 = 1;
    let dst_col0 = 1;

    for row in 0..rect_rows {
        for col in 0..rect_cols {
            let src_idx = (src_row0 + row) * src_cols + (src_col0 + col);
            let dst_idx = (dst_row0 + row) * dst_cols + (dst_col0 + col);
            zero_matrix[dst_idx] = read_back[src_idx];
        }
    }

    for i in 0..8 {
        for j in 0..10 {
            print!("{:6.1}", zero_matrix[i * 10 + j]);
        }
        println!();
    }

    Ok(())
}