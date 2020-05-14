#include<pybind11/pybind11.h>
#include <dataset/dataset.hpp>

namespace py = pybind11;

using namespace dataset;

namespace util {

    void check_edge_list(const DataFrame& df, const std::vector<py::tuple>& list) {

        auto schema = df->schema();


        for (auto tup : list) {

            if (tup.size() != 2) {
                throw std::invalid_argument("A tuple with 2 elements must be provided to represent each edge.");
            }

            try {
                auto s = tup[0].cast<std::string>();
                if(!schema->GetFieldByName(s))
                    throw std::invalid_argument("Node " + s + " not present in the data set.");

                s = tup[1].cast<std::string>();
                if(!schema->GetFieldByName(s))
                    throw std::invalid_argument("Node " + s + " not present in the data set.");
            } catch(pybind11::cast_error) {
                throw std::invalid_argument("The names of the nodes could not be casted to string.");
            }
        }
    }
}