#include<pybind11/pybind11.h>
#include <dataset/dataset.hpp>
#include <graph/dag.hpp>

namespace py = pybind11;

using namespace dataset;

using graph::arc_vector;

namespace util {

    arc_vector check_edge_list(const DataFrame& df, const std::vector<py::tuple>& list) {

        auto schema = df->schema();

        arc_vector res;
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
}