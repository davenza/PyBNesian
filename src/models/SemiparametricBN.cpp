#include <models/SemiparametricBN.hpp>

namespace models {

    py::tuple SemiparametricBN::__getstate_extra__() const {
        std::vector<int> state;

        for (size_t i = 0; i < this->physical_num_nodes(); ++i) {
            if (is_valid(i))
                state.push_back(static_cast<int>(this->node_type(i)));
        }

        return py::make_tuple(state);
    }

    void SemiparametricBN::__setstate_extra__(py::tuple& t) {
        std::vector<int> new_state = t[0].cast<std::vector<int>>();

        for (size_t i = 0; i < new_state.size(); ++i) {
            this->set_node_type(i, FactorType(static_cast<uint8_t>(new_state[i])));
        }
    }

    void SemiparametricBN::__setstate_extra__(py::tuple&& t) {
        __setstate_extra__(t);
    }

    py::tuple ConditionalSemiparametricBN::__getstate_extra__() const {
        std::vector<int> state;

        for (int i = 0; i < this->num_nodes(); ++i) {
            state.push_back(static_cast<int>(this->node_type(i)));
        }

        return py::make_tuple(state);
    }

    void ConditionalSemiparametricBN::__setstate_extra__(py::tuple& t) {
        std::vector<int> new_state = t[0].cast<std::vector<int>>();

        for (size_t i = 0; i < new_state.size(); ++i) {
            this->set_node_type(i, FactorType(static_cast<uint8_t>(new_state[i])));
        }
    }

    void ConditionalSemiparametricBN::__setstate_extra__(py::tuple&& t) {
        __setstate_extra__(t);
    }
}