use ocl::{ProQue, Buffer,Result};

fn main() -> Result<()> {
    let kernel_src = r#"
        __kernel void matvec_mult(__global float* mat, __global float* vec, __global float* result) {
            int gid = get_global_id(0);
            result[gid] = 0.0f;
            for (int i = 0; i < 4; i++) {
                result[gid] += mat[gid * 4 + i] * vec[i];
            }
        }
    "#;
    let mut mat = [0.0f32; 16];
    let mut vec = [0.0f32; 4];
    let mut result = [0.0f32; 4];
    let mut correct = [0.0f32; 4];
    // Initialize data
    for i in 0..16 {
        mat[i] = i as f32 * 2.0;
    }
    for i in 0..4 {
        vec[i] = i as f32 * 3.0;
        for j in 0..4 {correct[j] += mat[i+4*j] * vec[i];}
    }
    let pro_que = ProQue::builder().src(kernel_src).dims(4).build().unwrap();
    let mat_buf = Buffer::<f32>::builder().queue(pro_que.queue().clone()).flags(ocl::flags::MEM_READ_ONLY).len(16).copy_host_slice(&mat).build().unwrap();
    let vec_buf = Buffer::<f32>::builder().queue(pro_que.queue().clone()).flags(ocl::flags::MEM_READ_ONLY).len(4).copy_host_slice(&vec).build().unwrap();
    let res_buf = Buffer::<f32>::builder().queue(pro_que.queue().clone()).flags(ocl::flags::MEM_WRITE_ONLY).len(4).build().unwrap();
    let kernel = pro_que.kernel_builder("matvec_mult").arg(&mat_buf).arg(&vec_buf).arg(&res_buf).build().unwrap();
    for _i in 0..4 {
        use std::time::Instant;
        let now = Instant::now();
        unsafe {kernel.cmd().global_work_size(4).local_work_size(4).enq().unwrap();}
        let _=pro_que.finish()?;
        let elapsed = now.elapsed();
        println!("Elapsed: {:.2?}", elapsed);
        res_buf.read(&mut result[..]).enq().unwrap();
    }
    let epsilon = 1e-5;
    let mut success = true;
    for i in 0..4 {
        if (result[i] - correct[i]).abs() > epsilon {
            success = false;
            break;
        }
    }
    if success {
        println!("Matrix-vector multiplication successful.");
    } else {
        println!("Matrix-vector multiplication unsuccessful.");
        println!("Result:  {:?}", result);
        println!("Correct: {:?}", correct);
    }
    Ok(())
}