use ocl::{ProQue, flags::MemFlags};

fn main() -> ocl::Result<()> {
    let vector_size = 1024;
    let alpha = 2.0f32;

    // Initialize vectors
    let a: Vec<f32> = (0..vector_size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..vector_size).map(|i| (vector_size - i) as f32).collect();
    let mut c = vec![0.0f32; vector_size];

    // OpenCL kernel source
    let kernel_source = r#"
        __kernel void saxpy_kernel(float alpha,
                                   __global const float *A,
                                   __global const float *B,
                                   __global float *C) {
            int index = get_global_id(0);
            C[index] = alpha * A[index] + B[index];
        }
    "#;

    // Initialize OpenCL environment
    let pro_que = ProQue::builder()
        .src(kernel_source)
        .dims(vector_size)
        .build()?;

    // Create buffers
    let a_buffer = pro_que.buffer_builder().flags(MemFlags::READ_ONLY | MemFlags::COPY_HOST_PTR).copy_host_slice(&a).build()?;
    let b_buffer = pro_que.buffer_builder().flags(MemFlags::READ_ONLY | MemFlags::COPY_HOST_PTR).copy_host_slice(&b).build()?;
    let c_buffer = pro_que.buffer_builder().flags(MemFlags::WRITE_ONLY).len(vector_size).build()?;

    // Create and execute kernel
    let kernel = pro_que.kernel_builder("saxpy_kernel")
        .arg(alpha)
        .arg(&a_buffer)
        .arg(&b_buffer)
        .arg(&c_buffer)
        .local_work_size(64)
        .build()?;

    let local_size = 64;
    unsafe {kernel.cmd().global_work_size(vector_size).local_work_size(local_size).enq()?;}

    // Read results
    c_buffer.read(&mut c).enq()?;
    for i in 0..vector_size {
        println!("{} * {} + {} = {}", alpha, a[i], b[i], c[i]);
    }

    Ok(())
}