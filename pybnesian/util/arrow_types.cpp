#include <util/arrow_types.hpp>

namespace util {

#define GET_PRIMITIVE_TYPE(NAME, FACTORY) \
    case Type::NAME:                      \
        return FACTORY()

std::shared_ptr<DataType> GetPrimitiveType(Type::type type) {
    switch (type) {
        case Type::NA:
            return arrow::null();
            GET_PRIMITIVE_TYPE(UINT8, arrow::uint8);
            GET_PRIMITIVE_TYPE(INT8, arrow::int8);
            GET_PRIMITIVE_TYPE(UINT16, arrow::uint16);
            GET_PRIMITIVE_TYPE(INT16, arrow::int16);
            GET_PRIMITIVE_TYPE(UINT32, arrow::uint32);
            GET_PRIMITIVE_TYPE(INT32, arrow::int32);
            GET_PRIMITIVE_TYPE(UINT64, arrow::uint64);
            GET_PRIMITIVE_TYPE(INT64, arrow::int64);
            GET_PRIMITIVE_TYPE(DATE32, arrow::date32);
            GET_PRIMITIVE_TYPE(DATE64, arrow::date64);
            GET_PRIMITIVE_TYPE(BOOL, arrow::boolean);
            GET_PRIMITIVE_TYPE(HALF_FLOAT, arrow::float16);
            GET_PRIMITIVE_TYPE(FLOAT, arrow::float32);
            GET_PRIMITIVE_TYPE(DOUBLE, arrow::float64);
            GET_PRIMITIVE_TYPE(BINARY, arrow::binary);
            GET_PRIMITIVE_TYPE(STRING, arrow::utf8);
            GET_PRIMITIVE_TYPE(LARGE_BINARY, arrow::large_binary);
            GET_PRIMITIVE_TYPE(LARGE_STRING, arrow::large_utf8);
            GET_PRIMITIVE_TYPE(INTERVAL_MONTH_DAY_NANO, arrow::month_day_nano_interval);
        default:
            return nullptr;
    }
}

}  // namespace util