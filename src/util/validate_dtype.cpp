#include <util/validate_dtype.hpp>
#include <pybind11/pybind11.h>
#include <dataset/dataset.hpp>
#include <graph/dag.hpp>

namespace py = pybind11;

using namespace dataset;
using util::ArcVector, util::FactorTypeVector;

namespace util {

    ArcVector check_edge_list(const DataFrame& df, const std::vector<py::tuple>& list) {
        auto schema = df->schema();

        ArcVector res;
        res.reserve(list.size());

        for (auto tup : list) {

            if (tup.size() != 2) {
                throw std::invalid_argument("A tuple with 2 elements must be provided to represent each edge.");
            }

            try {
                auto s = tup[0].cast<std::string>();
                if(!schema->GetFieldByName(s))
                    throw std::invalid_argument("Node " + s + " not present in the data set.");

                auto d = tup[1].cast<std::string>();
                if(!schema->GetFieldByName(s))
                    throw std::invalid_argument("Node " + s + " not present in the data set.");
                    
                res.push_back(std::pair(std::move(s), std::move(d)));
            } catch(pybind11::cast_error) {
                throw std::invalid_argument("The names of the nodes could not be casted to string.");
            }
        }

        return res;
    }

    FactorTypeVector check_node_type_list(const DataFrame& df, const std::vector<py::tuple>& list) {
        auto schema = df->schema();

        FactorTypeVector res;
        res.reserve(list.size());

        for (auto tup : list) {

            if (tup.size() != 2) {
                throw std::invalid_argument("A tuple with 2 elements (name, node_type) must be provided to represent each node.");
            }

            try {
                auto s = tup[0].cast<std::string>();
                if(!schema->GetFieldByName(s))
                    throw std::invalid_argument("Node " + s + " not present in the data set.");

                auto n = tup[1].cast<std::string>();
                if (n == "LG") res.push_back(std::pair(std::move(s), FactorType::LinearGaussianCPD));
                else if (n == "CKDE") res.push_back(std::pair(std::move(s), FactorType::CKDE));
                else
                    throw std::invalid_argument("Wrong node type \"" + n + "\"specified. The possible alternatives are " 
                                "\"LG\" (Linear Gaussian CPD) or \"CKDE\" (Conditional KDE CPD).");
            } catch(pybind11::cast_error) {
                throw std::invalid_argument("The names of the nodes could not be casted to string.");
            }
        }

        return res;
    }
}