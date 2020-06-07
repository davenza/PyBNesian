#ifndef PGM_OPENCL_CONFIG_HPP
#define PGM_OPENCL_CONFIG_HPP

#define CL_HPP_ENABLE_EXCEPTIONS
// #define CL_HPP_MINIMUM_OPENCL_VERSION 120
// #define CL_HPP_TARGET_OPENCL_VERSION 120

#include <CL/cl2.hpp>
#include <fstream>

namespace opencl {

    inline constexpr int default_platform_idx = 0;
    inline constexpr int default_device_idx = 0;

    class OpenCLConfig {
    public:
        static OpenCLConfig& get();

        cl::Context& context() { return m_context; }
        cl::Program& program() { return m_program; }

        template<typename T>
        cl::Buffer copy_to_buffer(const T* d, int size);
    private:
        OpenCLConfig() {}
        OpenCLConfig(cl::Context cont, cl::CommandQueue queue, cl::Program program) : m_context(cont), 
                                                                                      m_queue(queue), 
                                                                                      m_program(program) {}

        static OpenCLConfig singleton;
        static bool initialized;

        cl::Context m_context;
        cl::CommandQueue m_queue;
        cl::Program m_program;
    };



    template<typename T>
    cl::Buffer OpenCLConfig::copy_to_buffer(const T* d, int size) {
        cl_int err_code = CL_SUCCESS;
        cl::Buffer b(m_context, CL_MEM_READ_WRITE,  sizeof(T)*size, NULL, &err_code);

        if (err_code != CL_SUCCESS) {
            throw std::runtime_error("Error creating OpenCL buffer.");
        }

        err_code = m_queue.enqueueWriteBuffer(b, CL_TRUE, 0, sizeof(T)*size, d);

        if (err_code != CL_SUCCESS) {
            throw std::runtime_error("Error copying OpenCL buffer.");
        }

        return std::move(b);
    }
}

// #define CL_HPP_ENABLE_EXCEPTIONS
// #define CL_HPP_TARGET_OPENCL_VERSION 120

#endif //PGM_OPENCL_CONFIG_HPP