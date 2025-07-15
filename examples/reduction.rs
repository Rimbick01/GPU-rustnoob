use ocl::{Platform, Device, Context, Queue, Program, Buffer, flags, DeviceType, builders::KernelBuilder};

const ARRAY_SIZE: usize = 65536;
const NUM_KERNELS: usize = 2;

fn main() -> ocl::Result<()> {
    let src = std::fs::read_to_string("hello_kernel.cl").expect("Failed to read reduction.cl");

    let platform = Platform::list().into_iter().next().unwrap();
    let mut devices = Device::list(platform, Some(DeviceType::GPU)).unwrap_or_default();
    if devices.is_empty() {
        devices = Device::list(platform, Some(DeviceType::CPU)).unwrap_or_default();
    }
    let device = devices.into_iter().next().unwrap();

    let context = Context::builder().platform(platform).devices(device.clone()).build()?;
    let queue = Queue::new(&context, device.clone(), Some(flags::QUEUE_PROFILING_ENABLE))?;

    let local_size = 128usize;
    let global_size_scalar = ARRAY_SIZE;
    let global_size_vector = ARRAY_SIZE / 4;

    let mut data = vec![0.0f32; ARRAY_SIZE];
    for i in 0..ARRAY_SIZE {
        data[i] = i as f32;
    }

    let data_buffer = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .len(ARRAY_SIZE)
        .copy_host_slice(&data)
        .build()?;

    let program = Program::builder()
        .src(src)
        .devices(device.clone())
        .build(&context)?;

    let kernel_names = ["reduction_scalar", "reduction_vector"];

    for i in 0..NUM_KERNELS {
        let (global_size, local_mem_size, num_groups) = if i == 0 {
            let num_groups = global_size_scalar / local_size;
            (global_size_scalar, local_size, num_groups)
        } else {
            let num_groups = global_size_vector / local_size;
            (global_size_vector, local_size * 4, num_groups)
        };

        let sums_buffer = Buffer::<f32>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_WRITE_ONLY)
            .len(num_groups)
            .build()?;

        let kernel = KernelBuilder::new()
            .program(&program)
            .name(kernel_names[i])
            .queue(queue.clone())
            .arg(&data_buffer)
            .arg_local::<f32>(local_mem_size)
            .arg(&sums_buffer)
            .global_work_size(global_size)
            .local_work_size(local_size)
            .build()?;


        unsafe {
            kernel.cmd()
                .queue(&queue)
                .global_work_size(global_size)
                .local_work_size(local_size)
                .enq()?;
        }

        queue.finish()?;

        let mut sums = vec![0.0f32; num_groups];
        sums_buffer.read(&mut sums[..]).enq()?;
        let sum: f32 = sums.iter().sum();

        println!("{} sum is: {}", kernel_names[i], sum);
        let actual_sum = (ARRAY_SIZE as f32 / 2.0) * ((ARRAY_SIZE - 1) as f32);
        if (sum - actual_sum).abs() > 0.01 * sum.abs() {
            println!("Check failed.");
        } else {
            println!("Check passed.");
        }
    }

    Ok(())
}