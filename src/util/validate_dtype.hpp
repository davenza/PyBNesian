#ifndef PYBNESIAN_UTIL_VALIDATE_DTYPE_HPP
#define PYBNESIAN_UTIL_VALIDATE_DTYPE_HPP

#include <arrow/api.h>

namespace util {

    std::shared_ptr<arrow::DataType> to_type(arrow::Type::type t);
}

#endif //PYBNESIAN_UTIL_VALIDATE_DTYPE_HPP