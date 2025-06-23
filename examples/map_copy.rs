use cl_sys::*;
use std::ffi::c_void;
use std::mem;
use std::ptr;

fn main() {
    unsafe {
        let mut platform: cl_platform_id = ptr::null_mut();
        clGetPlatformIDs(1, &mut platform, ptr::null_mut());

        let mut device: cl_device_id = ptr::null_mut();
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &mut device, ptr::null_mut());

        let mut err: cl_int = 0;
        let context = clCreateContext(ptr::null(), 1, &device, None, ptr::null_mut(), &mut err);
        assert_eq!(err, CL_SUCCESS);

        let queue = clCreateCommandQueue(context, device, 0, &mut err);
        assert_eq!(err, CL_SUCCESS);

        let mut data_1 = [0.0f32; 100];
        let mut data_2 = [0.0f32; 100];
        let mut result = [0.0f32; 100];
        for i in 0..100 {
            data_1[i] = i as f32;
            data_2[i] = 0.0;
            result[i] = 0.0;
        }

        let buffer_1 = clCreateBuffer(context,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,mem::size_of_val(&data_1),data_1.as_ptr() as *mut c_void,&mut err,);
        assert_eq!(err, CL_SUCCESS);

        let buffer_2 = clCreateBuffer(context,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,mem::size_of_val(&data_2),data_2.as_ptr() as *mut c_void,&mut err,);
        assert_eq!(err, CL_SUCCESS);

        err = clEnqueueCopyBuffer(queue,buffer_1,buffer_2,0,0,mem::size_of_val(&data_1),0,ptr::null(),ptr::null_mut());
        assert_eq!(err, CL_SUCCESS);

        let mut map_err: cl_int = 0;
        let mapped_memory = clEnqueueMapBuffer(queue,buffer_2,CL_TRUE as cl_bool,CL_MAP_READ,0,mem::size_of_val(&data_2),0,ptr::null(),ptr::null_mut(),&mut map_err);
        assert_eq!(map_err, CL_SUCCESS);

        let mapped_slice = std::slice::from_raw_parts(mapped_memory as *const f32, 100);
        result.copy_from_slice(mapped_slice);

        err = clEnqueueUnmapMemObject(queue,buffer_2,mapped_memory,0,ptr::null(),ptr::null_mut());
        assert_eq!(err, CL_SUCCESS);

        for i in 0..10 {
            for j in 0..10 {
                print!("{:6.1}", result[i * 10 + j]);
            }
            println!();
        }

        clReleaseMemObject(buffer_1);
        clReleaseMemObject(buffer_2);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        clReleaseDevice(device);
    }
}