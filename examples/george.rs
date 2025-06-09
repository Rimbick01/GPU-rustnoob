use std::{cmp::max, cmp::min, error::Error};
use plotters::prelude::*;
use ocl::{ProQue, Buffer, MemFlags};


fn main() -> Result<(), Box<dyn Error>> {
    let kernel_src = r#"
        __kernel void add(
            __global float* c
        ) {
            float a = get_local_id(0);
            for (int i = 0; i < 1000000; i++) { a *= 2; }
            c[get_global_id(0)] = get_local_id(0);
            c[get_global_id(0)+128] = a;
        }
    "#;

    // Initialize ProQue (Program, Queue, Context)
    let proque = ProQue::builder()
        .src(kernel_src)
        .dims(1024)
        .build()?;

    let c_buffer = Buffer::builder().queue(proque.queue().clone()).flags(MemFlags::new().read_write()).len(32768).build()?;

    // Build kernel and set arguments
    let kernel = proque.kernel_builder("add")
        .arg(&c_buffer)
        .build()?;

    // warmup
    for _warmup in 0..2 {
        unsafe { kernel.cmd().global_work_size(1).local_work_size(1).enq()?; }
        proque.finish()?;
    }

    // draw plot
    let root = BitMapBackend::new("plot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Based Line Plot", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..40960, 0..50000)?;

    chart.configure_mesh().draw()?;

    for (locals, color) in [(1, RED), (2, MAGENTA), (4, CYAN), (8, RED), (16, MAGENTA), (32, GREEN), (64, BLUE)] {
        let mut points = Vec::new();

        for test_cores in (64..min(40960, 640*2*locals)).step_by(max(64, locals)) {
            use std::time::Instant;
            let now = Instant::now();

            unsafe { kernel.cmd().global_work_size(test_cores).local_work_size(locals).enq()?; }

            // Wait for compute
            proque.finish()?;
            let elapsed = now.elapsed();
            println!("Elapsed: {:.2?} for {test_cores} with {locals}", elapsed);
            points.push((test_cores as i32, elapsed.as_micros() as i32));
        }
        chart.draw_series(LineSeries::new(points, &color))?;
    }

    // Read result back
    let mut c_data = vec![0.0f32; 128];
    c_buffer.cmd().read(&mut c_data).enq()?;

    // Verify output
    let mut i = 0;
    for &c in &c_data {
        if i % 16 == 0 && i != 0 {
            println!("");
        }
        i += 1;
        print!("{:>3} ", c);
        if i == 128 { break; }
    }
    println!("");

    Ok(())
}

