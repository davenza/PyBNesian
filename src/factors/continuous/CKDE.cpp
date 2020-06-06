#include <factors/continuous/CKDE.hpp>
#include <opencl/opencl_config.hpp>


using opencl::OpenCLConfig;

namespace factors::continuous {


    template<typename ArrowType>
    void CKDE::_fit(const DataFrame& df) {

    }

    template<typename ArrowType>
    void CKDE::_fit_null(const DataFrame& df) {
        
    }



    void CKDE::fit(py::handle pyobject) {
        auto rb = dataset::to_record_batch(pyobject);
        auto df = DataFrame(rb);
        fit(df);
    }

    void CKDE::fit(const DataFrame& df) {
        auto type_id = df.same_type(m_variable, m_evidence);

        bool contains_null = df.null_count(m_variable, m_evidence);

        switch(type_id) {
            case Type::DOUBLE: {
                if (contains_null)
                    _fit_null<arrow::DoubleType>(df);
                else 
                    _fit<arrow::DoubleType>(df);
                break;
            }
            case Type::FLOAT: {
                if (contains_null)
                    _fit_null<arrow::FloatType>(df);
                else 
                    _fit<arrow::FloatType>(df);
                break;
            }
            default:
                throw py::value_error("Wrong data type to fit CKDE. [double] or [float] data is expected.");
        }
    }


    void opencl() {
        //get all platforms (drivers)
        std::vector<cl::Platform> all_platforms;
        cl::Platform::get(&all_platforms);
        if(all_platforms.size()==0){
            std::cout<<" No platforms found. Check OpenCL installation!\n";
            // exit(1);
        }
        std::cout << all_platforms.size() << " platforms found." << std::endl;

        for (auto platform : all_platforms) {
            std::cout << platform.getInfo<CL_PLATFORM_NAME>() << ". Version: " 
                        << platform.getInfo<CL_PLATFORM_VERSION>() << std::endl; 
        }

        auto cl_config = OpenCLConfig::init_opencl();
    }
}