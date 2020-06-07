#include <iostream>
#include <opencl/opencl_config.hpp>


namespace opencl {

        OpenCLConfig& OpenCLConfig::get() {

        if (initialized)
            return singleton;

        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            throw std::runtime_error("OpenCL platforms not found.");
        }

        cl::Platform plat = cl::Platform::setDefault(platforms[default_platform_idx]);
        if (plat != platforms[default_platform_idx]) {
            throw std::runtime_error("Error setting default platform.");
        }

        std::vector<cl::Device> devices;
        plat.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if(devices.size()==0){
            throw std::runtime_error("No devices found. Check OpenCL installation!");
        }

        cl::Device dev = cl::Device::setDefault(devices[default_device_idx]);

        if (dev != devices[default_device_idx]) {
            throw std::runtime_error("Error setting default device.");
        }

        cl::Context context(dev);

        cl::CommandQueue queue(context, dev);

        // Read the program source
        std::ifstream sourceFile("src/factors/continuous/opencl/CKDE.cl");
        std::string sourceCode( std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source({sourceCode});

        cl::Program program (context, source);

        cl_int build_result = program.build();

        if (build_result != CL_SUCCESS) {
            cl_int buildErr = CL_SUCCESS;
            auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
            for (auto &pair : buildInfo) {
                std::cerr << pair.second << std::endl << std::endl;
            }
            throw std::runtime_error("Error compilating OpenCL code.");
        }

        OpenCLConfig::singleton =  OpenCLConfig(context, queue, program);
        OpenCLConfig::initialized = true;

        return OpenCLConfig::singleton;
    }
}