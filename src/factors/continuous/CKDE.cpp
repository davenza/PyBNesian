#include <factors/continuous/CKDE.hpp>

namespace factors::continuous {



    void CKDE::fit(py::handle pyobject) {
        auto rb = dataset::to_record_batch(pyobject);
        auto df = DataFrame(rb);
        fit(df);
    }

    void CKDE::fit(const DataFrame& df) {

    }


    void opencl() {
        //get all platforms (drivers)
        std::vector<cl::Platform> all_platforms;
        cl::Platform::get(&all_platforms);
        if(all_platforms.size()==0){
            std::cout<<" No platforms found. Check OpenCL installation!\n";
            // exit(1);
        }
        cl::Platform default_platform=all_platforms[0];
        std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";
    }
}