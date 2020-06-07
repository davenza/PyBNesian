#ifndef PGM_OPENCL_CONFIG_HPP
#define PGM_OPENCL_CONFIG_HPP

#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/cl2.hpp>
#include <fstream>

namespace opencl {

    inline constexpr int default_platform_idx = 0;
    inline constexpr int default_device_idx = 0;

    class OpenCLConfig {
    public:
        static OpenCLConfig& get();
    private:
        static OpenCLConfig singleton;
        static bool initialized;
    };
}

#endif PGM_OPENCL_CONFIG_HPP