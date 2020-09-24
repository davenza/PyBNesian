#include <iostream>
#include <opencl/opencl_config.hpp>

namespace opencl {


    const char* opencl_error(cl_int error) {
        switch(error){
            // run-time and JIT compiler errors
            case 0: return "CL_SUCCESS";
            case -1: return "CL_DEVICE_NOT_FOUND";
            case -2: return "CL_DEVICE_NOT_AVAILABLE";
            case -3: return "CL_COMPILER_NOT_AVAILABLE";
            case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
            case -5: return "CL_OUT_OF_RESOURCES";
            case -6: return "CL_OUT_OF_HOST_MEMORY";
            case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
            case -8: return "CL_MEM_COPY_OVERLAP";
            case -9: return "CL_IMAGE_FORMAT_MISMATCH";
            case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
            case -11: return "CL_BUILD_PROGRAM_FAILURE";
            case -12: return "CL_MAP_FAILURE";
            case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
            case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
            case -15: return "CL_COMPILE_PROGRAM_FAILURE";
            case -16: return "CL_LINKER_NOT_AVAILABLE";
            case -17: return "CL_LINK_PROGRAM_FAILURE";
            case -18: return "CL_DEVICE_PARTITION_FAILED";
            case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

            // compile-time errors
            case -30: return "CL_INVALID_VALUE";
            case -31: return "CL_INVALID_DEVICE_TYPE";
            case -32: return "CL_INVALID_PLATFORM";
            case -33: return "CL_INVALID_DEVICE";
            case -34: return "CL_INVALID_CONTEXT";
            case -35: return "CL_INVALID_QUEUE_PROPERTIES";
            case -36: return "CL_INVALID_COMMAND_QUEUE";
            case -37: return "CL_INVALID_HOST_PTR";
            case -38: return "CL_INVALID_MEM_OBJECT";
            case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
            case -40: return "CL_INVALID_IMAGE_SIZE";
            case -41: return "CL_INVALID_SAMPLER";
            case -42: return "CL_INVALID_BINARY";
            case -43: return "CL_INVALID_BUILD_OPTIONS";
            case -44: return "CL_INVALID_PROGRAM";
            case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
            case -46: return "CL_INVALID_KERNEL_NAME";
            case -47: return "CL_INVALID_KERNEL_DEFINITION";
            case -48: return "CL_INVALID_KERNEL";
            case -49: return "CL_INVALID_ARG_INDEX";
            case -50: return "CL_INVALID_ARG_VALUE";
            case -51: return "CL_INVALID_ARG_SIZE";
            case -52: return "CL_INVALID_KERNEL_ARGS";
            case -53: return "CL_INVALID_WORK_DIMENSION";
            case -54: return "CL_INVALID_WORK_GROUP_SIZE";
            case -55: return "CL_INVALID_WORK_ITEM_SIZE";
            case -56: return "CL_INVALID_GLOBAL_OFFSET";
            case -57: return "CL_INVALID_EVENT_WAIT_LIST";
            case -58: return "CL_INVALID_EVENT";
            case -59: return "CL_INVALID_OPERATION";
            case -60: return "CL_INVALID_GL_OBJECT";
            case -61: return "CL_INVALID_BUFFER_SIZE";
            case -62: return "CL_INVALID_MIP_LEVEL";
            case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
            case -64: return "CL_INVALID_PROPERTY";
            case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
            case -66: return "CL_INVALID_COMPILER_OPTIONS";
            case -67: return "CL_INVALID_LINKER_OPTIONS";
            case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

            // extension errors
            case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
            case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
            case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
            case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
            case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
            case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
            default: return "Unknown OpenCL error";
        }
    }


    // bool OpenCLConfig::initialized = false;
    
    OpenCLConfig::OpenCLConfig() {
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

        try{
            program.build();
        } catch(...) {
            cl_int buildErr = CL_SUCCESS;
            auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
            for (auto &pair : buildInfo) {
                std::cerr << pair.second << std::endl << std::endl;
            }
            throw std::runtime_error("Error compilating OpenCL code.");
        }

        cl_int err_code = CL_SUCCESS;
        int max_local_size = dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&err_code);
        if (err_code != CL_SUCCESS) {
            throw std::runtime_error(std::string("Maximum work group size could not be determined. ") + 
                                     opencl_error(err_code) + " (" + std::to_string(err_code) + ").");
        }

        m_context = context;
        m_queue = queue;
        m_program = program;
        m_device = dev;
        m_max_local_size = max_local_size;
    }

    OpenCLConfig& OpenCLConfig::get() {
        static OpenCLConfig singleton;
        return singleton;
    }

    cl::Kernel& OpenCLConfig::kernel(const char* name) {

        auto it = m_kernels.find(name);

        if (it != m_kernels.end()) {
            return it->second;
        } else {
            cl_int err_code = CL_SUCCESS;
            auto k = cl::Kernel(m_program, name, &err_code);
            if (err_code != CL_SUCCESS) {
                throw std::runtime_error(std::string("Error ") + opencl_error(err_code) + 
                                "(" + std::to_string(err_code) + ") creating OpenCL kernel " + name);
            }

            m_kernels.insert({name, std::move(k)});
            return m_kernels.find(name)->second;;
        }
    }



    void update_reduction_status(int& length, int& num_groups, int& local_size, int& global_size, int max_local_size) {
        length = num_groups;
        num_groups = static_cast<int>(std::ceil(static_cast<double>(length) / static_cast<double>(max_local_size)));
        local_size = (length > max_local_size) ? max_local_size : length;
        global_size = local_size * num_groups;
    }
}