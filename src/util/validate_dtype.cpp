#include <util/validate_dtype.hpp>
#include <pybind11/pybind11.h>
#include <dataset/dataset.hpp>
#include <graph/dag.hpp>

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
}