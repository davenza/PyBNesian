#ifndef PYBNESIAN_UTIL_ARROW_TYPES_HPP
#define PYBNESIAN_UTIL_ARROW_TYPES_HPP

#include <arrow/api.h>

using arrow::DataType, arrow::Type;

namespace util {

std::shared_ptr<DataType> GetPrimitiveType(Type::type type);

}

#endif  // PYBNESIAN_UTIL_ARROW_TYPES_HPP