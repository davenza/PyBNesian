#include <util/validate_dtype.hpp>
#include <pybind11/pybind11.h>
#include <dataset/dataset.hpp>

namespace py = pybind11;

using namespace dataset;
using util::ArcVector, util::FactorTypeVector;

namespace util {

    void check_edge_list(const DataFrame& df, const ArcVector& list) {
        auto schema = df->schema();

        for (auto pair : list) {
            if(!schema->GetFieldByName(pair.first))
                throw std::invalid_argument("Node " + pair.first + " not present in the data set.");

            if(!schema->GetFieldByName(pair.second))
                throw std::invalid_argument("Node " + pair.second + " not present in the data set.");
        }
    }

    void check_node_type_list(const DataFrame& df, const FactorTypeVector& list) {
        auto schema = df->schema();

        for (auto pair : list) {
            if(!schema->GetFieldByName(pair.first))
                throw std::invalid_argument("Node " + pair.first + " not present in the data set.");
        }
    }

    std::shared_ptr<arrow::DataType> to_type(arrow::Type::type t) {
        switch (t) {
            case Type::DOUBLE:
                return arrow::float64();
            case Type::FLOAT:
                return arrow::float32();
            case Type::HALF_FLOAT:
                return arrow::float16();
            case Type::INT8:
                return arrow::int8();
            case Type::INT16:
                return arrow::int16();
            case Type::INT32:
                return arrow::int32();
            case Type::INT64:
                return arrow::int64();
            case Type::UINT8:
                return arrow::uint8();
            case Type::UINT16:
                return arrow::uint16();
            case Type::UINT32:
                return arrow::uint32();
            case Type::UINT64:
                return arrow::uint64();
            case Type::BOOL:
                return arrow::boolean();
            case Type::NA:
                return arrow::null();
            case Type::STRING:
                return arrow::utf8();
            default:
                throw new std::invalid_argument("Data type not valid.");
        }
    }
}