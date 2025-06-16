use ocl::{ enums::{ProgramBuildInfo, ProgramBuildInfoResult}};
use ocl::builders::ProgramBuilder;
use ocl::{Context, Device, DeviceType,  Platform};
use std::fs;
use std::ffi::CString;



const PROGRAM_FILE: &str = "good.cl";
const PROGRAM_FILE_1: &str = "bad.cl";
const OPTIONS: &str = "-cl-finite-math-only -cl-no-signed-zeros";

fn main() -> ocl::Result<()> {

    let platforms = Platform::list();

    let platform = platforms.into_iter().next().expect("No platforms found");

    let mut devices = Device::list(platform, Some(DeviceType::GPU)).unwrap_or_default();
    if devices.is_empty() {
        devices = Device::list(platform, Some(DeviceType::CPU)).unwrap_or_default();
    }
    let dev = devices.into_iter().next().expect("No devices found");

    let context = Context::builder()
        .platform(platform)
        .devices(dev)
        .build()?;
        // .expect("Failed to create context");
    let file_names = [PROGRAM_FILE, PROGRAM_FILE_1];
    let mut program_sources = Vec::<String>::new();
    for file_name in &file_names {
        let program_handle = fs::read_to_string(file_name).unwrap_or_else(|_| panic!("Failed to read file: {}", file_name));
        program_sources.push(program_handle);
    }

    let program_con = ProgramBuilder::new()
        .src(program_sources[0].as_str())
        .src(program_sources[1].as_str())
        .devices(dev.clone())
        .cmplr_opt(&*OPTIONS)
        .build(&context)
        .unwrap(); 
    let options_cstring = CString::new(OPTIONS).unwrap();
    let build_result = ocl::core::build_program(&program_con, Some(&[dev.clone()]), &options_cstring,None,None);

    if build_result.is_err() {
        // Query and print the build log
        let log = match program_con.build_info(dev, ProgramBuildInfo::BuildLog) {
            Ok(ProgramBuildInfoResult::BuildLog(s)) => s,
            _ => "<No build log available>".to_string(),
        };
        println!("OpenCL build log:\n{}", log);
    }

    println!("Current directory: {:?}", std::env::current_dir());
    Ok(())
}