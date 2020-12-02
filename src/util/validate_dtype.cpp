#include <util/validate_dtype.hpp>

namespace util {

    std::shared_ptr<arrow::DataType> to_type(arrow::Type::type t) {
        switch (t) {
            case arrow::Type::DOUBLE:
                return arrow::float64();
            case arrow::Type::FLOAT:
                return arrow::float32();
            case arrow::Type::HALF_FLOAT:
                return arrow::float16();
            case arrow::Type::INT8:
                return arrow::int8();
            case arrow::Type::INT16:
                return arrow::int16();
            case arrow::Type::INT32:
                return arrow::int32();
            case arrow::Type::INT64:
                return arrow::int64();
            case arrow::Type::UINT8:
                return arrow::uint8();
            case arrow::Type::UINT16:
                return arrow::uint16();
            case arrow::Type::UINT32:
                return arrow::uint32();
            case arrow::Type::UINT64:
                return arrow::uint64();
            case arrow::Type::BOOL:
                return arrow::boolean();
            case arrow::Type::NA:
                return arrow::null();
            case arrow::Type::STRING:
                return arrow::utf8();
            default:
                throw new std::invalid_argument("Data type not valid.");
        }
    }
}