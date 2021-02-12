#include <pybind11/stl.h>
#include <dataset/crossvalidation_adaptator.hpp>
#include <dataset/holdout_adaptator.hpp>
#include <dataset/dynamic_dataset.hpp>

using dataset::DataFrame, dataset::CrossValidation, dataset::HoldOut, 
      dataset::DynamicDataFrame, dataset::DynamicVariable;

void pybindings_dataset(py::module& root) {
    auto dataset = root.def_submodule("dataset", "Dataset functionality.");

    py::class_<CrossValidation>(dataset, "CrossValidation")
        .def(py::init<DataFrame, int, bool>(), 
                    py::arg("df"), 
                    py::arg("k") = 10, 
                    py::arg("include_null") = false)
        .def(py::init<DataFrame, int, unsigned int, bool>(), 
                    py::arg("df"), 
                    py::arg("k") = 10, 
                    py::arg("seed"), 
                    py::arg("include_null") = false)
        .def("__iter__", [](CrossValidation& self) { 
                    return py::make_iterator(self.begin(), self.end()); }, py::keep_alive<0, 1>())
        .def("fold", &CrossValidation::fold, py::return_value_policy::take_ownership)
        .def("loc", [](CrossValidation& self, std::string name) { return self.loc(name); })
        .def("loc", [](CrossValidation& self, int idx) { return self.loc(idx); })
        .def("loc", [](CrossValidation& self, std::vector<std::string> v) { return self.loc(v); })
        .def("loc", [](CrossValidation& self, std::vector<int> v) { return self.loc(v); })
        .def("indices", [](CrossValidation& self) { 
                    return py::make_iterator(self.begin_indices(), self.end_indices()); }
                    );

    py::class_<HoldOut>(dataset, "HoldOut")
        .def(py::init<const DataFrame&, double, bool>(), 
                    py::arg("df"), 
                    py::arg("test_ratio") = 0.2, 
                    py::arg("include_null") = false)
        .def(py::init<const DataFrame&, double, unsigned int, bool>(), 
                    py::arg("df"), 
                    py::arg("test_ratio") = 0.2, 
                    py::arg("seed"), 
                    py::arg("include_null") = false)
        .def("training_data", &HoldOut::training_data, py::return_value_policy::reference_internal)
        .def("test_data", &HoldOut::test_data, py::return_value_policy::reference_internal);

    py::class_<DynamicVariable<int>>(dataset, "DynamicVariable<int>")
        .def(py::init<int, int>())
        .def(py::init<std::pair<int, int>>())
        .def_property("variable",
        [](DynamicVariable<int>& self) {
            return self.variable;
        },
        [](DynamicVariable<int>& self, int other) {
            self.variable = other;
        })
        .def_property("temporal_slice",
        [](DynamicVariable<int>& self) {
            return self.temporal_slice;
        },
        [](DynamicVariable<int>& self, int slice) {
            self.temporal_slice = slice;
        });
    
    py::class_<DynamicVariable<std::string>>(dataset, "DynamicVariable<std::string>")
        .def(py::init<std::string, int>())
        .def(py::init<std::pair<std::string, int>>())
        .def_property("variable",
        [](DynamicVariable<std::string>& self) {
            return self.variable;
        },
        [](DynamicVariable<std::string>& self, std::string other) {
            self.variable = other;
        })
        .def_property("temporal_slice",
        [](DynamicVariable<std::string>& self) {
            return self.temporal_slice;
        },
        [](DynamicVariable<std::string>& self, int slice) {
            self.temporal_slice = slice;
        });


    py::implicitly_convertible<std::pair<int, int>, DynamicVariable<int>>();
    py::implicitly_convertible<std::pair<std::string, int>, DynamicVariable<std::string>>();

    py::class_<DynamicDataFrame>(dataset, "DynamicDataFrame")
        .def(py::init<const DataFrame&, int>())
        .def("markovian_order", &DynamicDataFrame::markovian_order)
        .def("num_columns", &DynamicDataFrame::num_columns)
        .def("num_rows", &DynamicDataFrame::num_rows)
        .def("temporal_slice", [](const DynamicDataFrame& self, int slice_index) {
            return self.temporal_slice(slice_index);
        })
        .def("temporal_slice", [](const DynamicDataFrame& self, const std::vector<int>& slice_indices) {
            return self.temporal_slice(slice_indices.begin(), slice_indices.end());
        })
        .def("loc", [](const DynamicDataFrame& self, const DynamicVariable<int>& v) {
            return self.loc(v);
        })
        .def("loc", [](const DynamicDataFrame& self, const DynamicVariable<std::string>& v) {
            return self.loc(v);
        })
        .def("loc", [](const DynamicDataFrame& self, const std::vector<DynamicVariable<int>>& vec) {
            return self.loc(vec);
        })
        .def("loc", [](const DynamicDataFrame& self, const std::vector<DynamicVariable<std::string>>& vec) {
            return self.loc(vec);
        })
        .def("origin_df", &DynamicDataFrame::origin_df, py::return_value_policy::reference_internal)
        .def("static_df", &DynamicDataFrame::static_df, py::return_value_policy::reference_internal)
        .def("transition_df", &DynamicDataFrame::transition_df, py::return_value_policy::reference_internal);
}
