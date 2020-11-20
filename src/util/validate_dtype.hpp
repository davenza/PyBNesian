#ifndef PYBNESIAN_UTIL_VALIDATE_DTYPE_HPP
#define PYBNESIAN_UTIL_VALIDATE_DTYPE_HPP

#include <dataset/dataset.hpp>
#include <util/util_types.hpp>

namespace py = pybind11;

using namespace dataset;
using util::ArcStringVector, util::FactorStringTypeVector;

namespace util {
    
    void check_edge_list(const DataFrame& df, const ArcStringVector& list);
    void check_node_type_list(const DataFrame& df, const FactorStringTypeVector& list);

    std::shared_ptr<arrow::DataType> to_type(arrow::Type::type t); 


}

#endif //PYBNESIAN_UTIL_VALIDATE_DTYPE_HPP