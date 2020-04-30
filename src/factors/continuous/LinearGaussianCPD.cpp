#include <iostream>
#include <Python.h>
#include <arrow/api.h>
#include <arrow/python/pyarrow.h>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <dataset/dataset.hpp>
#include <linalg/linalg.hpp>

#include <arrow/compute/kernels/cast.h>
#include <arrow/compute/context.h>
#include <chrono>
#include <iomanip>

namespace py = pybind11;
namespace pyarrow = arrow::py;

using dataset::DataFrame;

namespace factors::continuous {

//    LinearGaussianCPD::LinearGaussianCPD() {};

    LinearGaussianCPD::LinearGaussianCPD(const std::string variable, const std::vector<std::string> evidence) :
    variable(variable),
    evidence(evidence)
    {
        this->beta.reserve(evidence.size() + 1);
    } ;

    LinearGaussianCPD::LinearGaussianCPD(const std::string variable, const std::vector<std::string> evidence,
            const std::vector<double> beta) :
    variable(variable),
    evidence(evidence),
    beta(beta)
//    TODO: Error checking
    {};

    void LinearGaussianCPD::fit(py::handle dataset) {
        auto rb = dataset::to_record_batch(dataset);
        auto df = DataFrame(rb);
//      TODO:  Check that columns are double/float.
        _fit(df);
    }

//    class FunctionContext;

    void LinearGaussianCPD::_fit(DataFrame df) {
        if (this->evidence.empty()) {
            this->beta[0] = linalg::mean(df.loc(this->variable));
            this->variance = linalg::var(df.loc(this->variable), this->beta[0]);
        } else if (this->evidence.size() == 1) {

        } else {

        }

//        auto column_array = df->GetColumnByName(this->variable);
//
//        auto fcontext = arrow::compute::FunctionContext();
//        auto cast_options = arrow::compute::CastOptions();
//
//        std::shared_ptr<arrow::Array> casted_array;
//        std::shared_ptr<arrow::DataType> to_type = std::make_shared<arrow::FloatType>();
//
//        std::cout << "Before cast: " << std::endl;
//        auto status = arrow::compute::Cast(&fcontext, *column_array, to_type, cast_options, &casted_array);
//
//        if(!status.ok()) {
//            std::cout << "Error: " << status.message() << std::endl;
//
//        } else {
//            std::cout << "Casted array: " << casted_array->ToString() << std::endl;
//            auto t1 = std::chrono::high_resolution_clock::now();
//            auto m = linalg::mean(casted_array, to_type);
//            auto t2 = std::chrono::high_resolution_clock::now();
//
//            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>( t2 - t1 ).count();
//
//            std::cout << "Mean casted: " << std::setprecision(24) << m << std::endl;
//            std::cout << "Nanoseconds: " << duration << std::endl;
//
//        }


    }
}