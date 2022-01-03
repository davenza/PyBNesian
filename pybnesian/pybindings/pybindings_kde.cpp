#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <kde/KDE.hpp>
#include <kde/ProductKDE.hpp>
#include <kde/BandwidthSelector.hpp>
#include <kde/ScottsBandwidth.hpp>
#include <kde/NormalReferenceRule.hpp>
#include <kde/UCV.hpp>
#include <util/exceptions.hpp>

using kde::KDE, kde::ProductKDE, kde::BandwidthSelector, kde::ScottsBandwidth, kde::NormalReferenceRule, kde::UCV,
    kde::UCVScorer;

using util::singular_covariance_data;

class PyBandwidthSelector : public BandwidthSelector {
public:
    bool is_python_derived() const override { return true; }

    VectorXd diag_bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const override {
        // PYBIND11_OVERRIDE_PURE(VectorXd, BandwidthSelector, diag_bandwidth, df, variables);
        pybind11::gil_scoped_acquire gil;
        pybind11::function override =
            pybind11::get_override(static_cast<const BandwidthSelector*>(this), "diag_bandwidth");

        if (override) {
            auto o = override(df, variables);

            try {
                auto m = o.cast<VectorXd>();

                if (static_cast<size_t>(m.rows()) != variables.size())
                    throw std::invalid_argument(
                        "BandwidthSelector::diag_bandwidth matrix must return a vector with shape "
                        "(" +
                        std::to_string(variables.size()) + ")");

                return m;
            } catch (py::cast_error& e) {
                throw std::runtime_error(
                    "The returned object of BandwidthSelector::diag_bandwidth is not a vector of doubles.");
            }
        }

        py::pybind11_fail("Tried to call pure virtual function \"BandwidthSelector::diag_bandwidth\"");
    }

    MatrixXd bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const override {
        // PYBIND11_OVERRIDE_PURE(MatrixXd, BandwidthSelector, bandwidth, df, variables);
        pybind11::gil_scoped_acquire gil;
        pybind11::function override = pybind11::get_override(static_cast<const BandwidthSelector*>(this), "bandwidth");

        if (override) {
            auto o = override(df, variables);

            try {
                auto m = o.cast<MatrixXd>();

                if (m.rows() != m.cols() || static_cast<size_t>(m.rows()) != variables.size())
                    throw std::invalid_argument(
                        "BandwidthSelector::bandwidth matrix must return an square matrix with shape "
                        "(" +
                        std::to_string(variables.size()) + ", " + std::to_string(variables.size()) + ")");

                return m;
            } catch (py::cast_error& e) {
                throw std::runtime_error(
                    "The returned object of BandwidthSelector::bandwidth is not a matrix of doubles.");
            }
        }

        py::pybind11_fail("Tried to call pure virtual function \"BandwidthSelector::bandwidth\"");
    }

    std::string ToString() const override {
        PYBIND11_OVERRIDE_PURE_NAME(std::string, BandwidthSelector, "__str__", ToString, );
    }

    py::tuple __getstate__() const override {
        py::gil_scoped_acquire gil;
        py::function override = py::get_override(static_cast<const BandwidthSelector*>(this), "__getstate_extra__");
        if (override) {
            return py::make_tuple(true, override());
        } else {
            return py::make_tuple(false, py::make_tuple());
        }
    }

    static void __setstate__(py::object& self, py::tuple& t) {
        // Call trampoline constructor
        py::gil_scoped_acquire gil;
        auto pyBandwidthSelector = py::type::of<BandwidthSelector>();
        pyBandwidthSelector.attr("__init__")(self);

        auto ptr = self.cast<const BandwidthSelector*>();

        auto extra_info = t[0].cast<bool>();
        if (extra_info) {
            py::function override = py::get_override(ptr, "__setstate_extra__");
            if (override) {
                override(t[1]);
            } else {
                py::pybind11_fail("Tried to call function \"BandwidthSelector::__setstate_extra__\"");
            }
        }
    }
};

void pybindings_kde(py::module& root) {
    //     py::exception<singular_covariance_data>(root, "SingularCovarianceData", PyExc_ValueError);
    py::register_exception<singular_covariance_data>(root, "SingularCovarianceData", PyExc_ValueError);

    py::class_<BandwidthSelector, PyBandwidthSelector, std::shared_ptr<BandwidthSelector>>(
        root, "BandwidthSelector", R"doc(
A :class:`BandwidthSelector <pybnesian.BandwidthSelector>` estimates the bandwidth of a kernel density estimation (KDE)
model.

If the bandwidth matrix cannot be calculated because the data has a singular covariance matrix, you should raise a
:class:`SingularCovarianceData <pybnesian.SingularCovarianceData>`.
)doc")
        .def(py::init<>(), R"doc(
Initializes a :class:`BandwidthSelector <pybnesian.BandwidthSelector>`.
)doc")
        .def("diag_bandwidth",
             &BandwidthSelector::diag_bandwidth,
             py::arg("df"),
             py::arg("variables"),
             R"doc(
Selects the bandwidth vector of a set of variables for a :class:`ProductKDE <pybnesian.ProductKDE>` with a given data
``df``.

:param df: DataFrame to select the bandwidth.
:param variables: A list of variables.
:returns: A numpy vector of floats. The i-th entry is the bandwidth :math:`h_{i}^{2}` for the ``variables[i]``.
)doc")
        .def("bandwidth",
             &BandwidthSelector::bandwidth,
             py::arg("df"),
             py::arg("variables"),
             R"doc(
Selects the bandwidth of a set of variables for a :class:`KDE <pybnesian.KDE>` with a given data ``df``.

:param df: DataFrame to select the bandwidth.
:param variables: A list of variables.
:returns: A float or numpy matrix of floats representing the bandwidth matrix.
)doc")
        .def("__getstate__", [](const BandwidthSelector& self) { return self.__getstate__(); })
        // Setstate for pyderived type
        .def("__setstate__", [](py::object& self, py::tuple& t) { PyBandwidthSelector::__setstate__(self, t); })
        .def("__repr__", [](const BandwidthSelector& self) { return self.ToString(); })
        .def("__str__", [](const BandwidthSelector& self) { return self.ToString(); });

    py::class_<ScottsBandwidth, BandwidthSelector, std::shared_ptr<ScottsBandwidth>>(root, "ScottsBandwidth", R"doc(
Selects the bandwidth using the Scott's rule [Scott]_:

.. math::

    \hat{h}_{i} = \hat{\sigma}_{i}\cdot N^{-1 / (d + 4)}.

This is a simplification of the normal reference rule.
)doc")
        .def(py::init<>(), R"doc(
Initializes a :class:`ScottsBandwidth <pybnesian.ScottsBandwidth>`.
)doc")
        .def(py::pickle([](const ScottsBandwidth& self) { return self.__getstate__(); },
                        [](py::tuple&) { return std::make_shared<ScottsBandwidth>(); }));

    py::class_<NormalReferenceRule, BandwidthSelector, std::shared_ptr<NormalReferenceRule>>(root,
                                                                                             "NormalReferenceRule",
                                                                                             R"doc(
Selects the bandwidth using the normal reference rule:

.. math::

    \hat{h}_{i} = \left(\frac{4}{d + 2}\right)^{1 / (d + 4)}\hat{\sigma}_{i}\cdot N^{-1 / (d + 4)}.

)doc")
        .def(py::init<>(), R"doc(
Initializes a :class:`NormalReferenceRule <pybnesian.NormalReferenceRule>`.
)doc")
        .def(py::pickle([](const NormalReferenceRule& self) { return self.__getstate__(); },
                        [](py::tuple&) { return std::make_shared<NormalReferenceRule>(); }));

    py::class_<UCVScorer>(root, "UCVScorer")
        .def(py::init<const DataFrame&, const std::vector<std::string>&>())
        .def("score_diagonal", &UCVScorer::score_diagonal)
        .def("score_unconstrained", &UCVScorer::score_unconstrained);

    py::class_<UCV, BandwidthSelector, std::shared_ptr<UCV>>(root, "UCV", R"doc(
Selects the bandwidth using the Unbiased Cross Validation (UCV) criterion (also known as least-squares cross
validation).

See Equation (3.8) in [MVKSA]_:

.. math::

    \text{UCV}(\mathbf{H}) = N^{-1}\lvert\mathbf{H}\rvert^{-1/2}(4\pi)^{-d/2} + \{N(N-1)\}^{-1}\sum\limits_{i, j:\ i \neq j}^{N}\{(1 - N^{-1})\phi_{2\mathbf{H}} - \phi_{\mathbf{H}}\}(\mathbf{t}_{i} - \mathbf{t}_{j})

where :math:`N` is the number of training instances, :math:`\phi_{\Sigma}` is the multivariate Gaussian kernel function
with covariance :math:`\Sigma`, :math:`\mathbf{t}_{i}` is the :math:`i`-th training instance, and :math:`\mathbf{H}` is
the bandwidth matrix.
)doc")
        .def(py::init<>(), R"doc(
Initializes a :class:`UCV <pybnesian.UCV>`.
)doc")
        .def(py::pickle([](const UCV& self) { return self.__getstate__(); },
                        [](py::tuple&) { return std::make_shared<UCV>(); }));

    py::class_<KDE>(root, "KDE", R"doc(
This class implements Kernel Density Estimation (KDE) for a set of variables:

.. math::

    \hat{f}(\text{variables}) = \frac{1}{N\lvert\mathbf{H} \rvert} \sum_{i=1}^{N}
    K(\mathbf{H}^{-1}(\text{variables} - \mathbf{t}_{i}))

where :math:`N` is the number of training instances, :math:`K()` is the multivariate Gaussian kernel function,
:math:`\mathbf{t}_{i}` is the :math:`i`-th training instance, and :math:`\mathbf{H}` is the bandwidth matrix.
)doc")
        .def(py::init<std::vector<std::string>>(), py::arg("variables"), R"doc(
Initializes a KDE with the given ``variables``. It uses the :class:`NormalReferenceRule <pybnesian.NormalReferenceRule>` as the default bandwidth
selector.

:param variables: List of variable names.
)doc")
        .def(py::init<>([](std::vector<std::string> variables, std::shared_ptr<BandwidthSelector> bandwidth_selector) {
                 return KDE(variables, BandwidthSelector::keep_python_alive(bandwidth_selector));
             }),
             py::arg("variables"),
             py::arg("bandwidth_selector"),
             R"doc(
Initializes a KDE with the given ``variables`` and ``bandwidth_selector`` procedure to fit the bandwidth.

:param variables: List of variable names.
:param bandwidth_selector: Procedure to fit the bandwidth.
)doc")
        .def("variables", &KDE::variables, R"doc(
Gets the variable names:

:returns: List of variable names.
)doc")
        .def("num_instances", &KDE::num_instances, R"doc(
Gets the number of training instances (:math:`N`).

:returns: Number of training instances.
)doc")
        .def("num_variables", &KDE::num_variables, R"doc(
Gets the number of variables.

:returns: Number of variables.
)doc")
        .def_property("bandwidth", &KDE::bandwidth, &KDE::setBandwidth, R"doc(
Bandwidth matrix (:math:`\mathbf{H}`)

)doc")
        .def("dataset", &KDE::training_data, R"doc(
Gets the training dataset for this KDE (the :math:`\mathbf{t}_{i}` instances).

:returns: Training instance.
)doc")
        .def("fitted", &KDE::fitted, R"doc(
Checks whether the model is fitted.

:returns: True if the model is fitted, False otherwise.
)doc")
        .def("data_type", &KDE::data_type, R"doc(
Returns the :class:`pyarrow.DataType` that represents the type of data handled by the :class:`KDE <pybnesian.KDE>`.

It can return :func:`pyarrow.float64 <pyarrow.float64>` or :func:`pyarrow.float32 <pyarrow.float32>`.

:returns: the :class:`pyarrow.DataType` physical data type representation of the :class:`KDE <pybnesian.KDE>`.
)doc")
        .def("fit", (void(KDE::*)(const DataFrame&)) & KDE::fit, py::arg("df"), R"doc(
Fits the :class:`KDE <pybnesian.KDE>` with the data in ``df``. It estimates the bandwidth :math:`\mathbf{H}` automatically using the
provided bandwidth selector.

:param df: DataFrame to fit the :class:`KDE <pybnesian.KDE>`.
)doc")
        .def("logl", &KDE::logl, py::return_value_policy::take_ownership, py::arg("df"), R"doc(
Returns the log-likelihood of each instance in the DataFrame ``df``.

:param df: DataFrame to compute the log-likelihood.
:returns: A :class:`numpy.ndarray` vector with dtype :class:`numpy.float64`, where the i-th value is the log-likelihod
          of the i-th instance of ``df``.
)doc")
        .def("slogl", &KDE::slogl, py::arg("df"), R"doc(
Returns the sum of the log-likelihood of each instance in the DataFrame ``df``. That is, the sum of the result of
:func:`KDE.logl <pybnesian.KDE.logl>`.

:param df: DataFrame to compute the sum of the log-likelihood.
:returns: The sum of log-likelihood for DataFrame ``df``.
)doc")
        .def("save", &KDE::save, py::arg("filename"), R"doc(
Saves the :class:`KDE <pybnesian.KDE>` in a pickle file with the given name.

:param filename: File name of the saved graph.
)doc")
        .def(py::pickle([](const KDE& self) { return self.__getstate__(); },
                        [](py::tuple t) { return KDE::__setstate__(t); }));

    py::class_<ProductKDE>(root, "ProductKDE", R"doc(
This class implements a product Kernel Density Estimation (KDE) for a set of variables:

.. math::

    \hat{f}(x_{1}, \ldots, x_{d}) = \frac{1}{N\cdot h_{1}\cdot\ldots\cdot h_{d}} \sum_{i=1}^{N}
    \prod_{j=1}^{d} K\left(\frac{(x_{j} - t_{ji})}{h_{j}}\right)

where :math:`N` is the number of training instances, :math:`d` is the dimensionality of the product KDE, :math:`K()` is
the multivariate Gaussian kernel function, :math:`t_{ji}` is the value of the :math:`j`-th variable in the
:math:`i`-th training instance, and :math:`h_{j}` is the bandwidth parameter for the :math:`j`-th variable.
)doc")
        .def(py::init<std::vector<std::string>>(), py::arg("variables"), R"doc(
Initializes a ProductKDE with the given ``variables``.

:param variables: List of variable names.
)doc")
        .def(py::init<>([](std::vector<std::string> variables, std::shared_ptr<BandwidthSelector> bandwidth_selector) {
                 return ProductKDE(variables, BandwidthSelector::keep_python_alive(bandwidth_selector));
             }),
             py::arg("variables"),
             py::arg("bandwidth_selector"),
             R"doc(
Initializes a ProductKDE with the given ``variables`` and ``bandwidth_selector`` procedure to fit the bandwidth.

:param variables: List of variable names.
:param bandwidth_selector: Procedure to fit the bandwidth.
)doc")
        .def("variables", &ProductKDE::variables, R"doc(
Gets the variable names:

:returns: List of variable names.
)doc")
        .def("num_instances", &ProductKDE::num_instances, R"doc(
Gets the number of training instances (:math:`N`).

:returns: Number of training instances.
)doc")
        .def("num_variables", &ProductKDE::num_variables, R"doc(
Gets the number of variables.

:returns: Number of variables.
)doc")
        .def_property("bandwidth", &ProductKDE::bandwidth, &ProductKDE::setBandwidth, R"doc(
Vector of bandwidth values (:math:`h_{j}^{2}`).
)doc")
        .def("dataset", &ProductKDE::training_data, R"doc(
Gets the training dataset for this ProductKDE (the :math:`\mathbf{t}_{i}` instances).

:returns: Training instance.
)doc")
        .def("fitted", &ProductKDE::fitted, R"doc(
Checks whether the model is fitted.

:returns: True if the model is fitted, False otherwise.
)doc")
        .def("data_type", &ProductKDE::data_type, R"doc(
Returns the :class:`pyarrow.DataType` that represents the type of data handled by the :class:`ProductKDE <pybnesian.ProductKDE>`.

It can return :func:`pyarrow.float64 <pyarrow.float64>` or :func:`pyarrow.float32 <pyarrow.float32>`.

:returns: the :class:`pyarrow.DataType` physical data type representation of the :class:`ProductKDE <pybnesian.ProductKDE>`.
)doc")
        .def("fit", (void(ProductKDE::*)(const DataFrame&)) & ProductKDE::fit, py::arg("df"), R"doc(
Fits the :class:`ProductKDE <pybnesian.ProductKDE>` with the data in ``df``. It estimates the bandwidth vector :math:`h_{j}` automatically
using the provided bandwidth selector.

:param df: DataFrame to fit the :class:`ProductKDE <pybnesian.ProductKDE>`.
)doc")
        .def("logl", &ProductKDE::logl, py::return_value_policy::take_ownership, py::arg("df"), R"doc(
Returns the log-likelihood of each instance in the DataFrame ``df``.

:param df: DataFrame to compute the log-likelihood.
:returns: A :class:`numpy.ndarray` vector with dtype :class:`numpy.float64`, where the i-th value is the log-likelihod
          of the i-th instance of ``df``.
)doc")
        .def("slogl", &ProductKDE::slogl, py::arg("df"), R"doc(
Returns the sum of the log-likelihood of each instance in the DataFrame ``df``. That is, the sum of the result of
:func:`ProductKDE.logl <pybnesian.ProductKDE.logl>`.

:param df: DataFrame to compute the sum of the log-likelihood.
:returns: The sum of log-likelihood for DataFrame ``df``.
)doc")
        .def("save", &ProductKDE::save, py::arg("filename"), R"doc(
Saves the :class:`ProductKDE <pybnesian.ProductKDE>` in a pickle file with the given name.

:param filename: File name of the saved graph.
)doc")
        .def(py::pickle([](const ProductKDE& self) { return self.__getstate__(); },
                        [](py::tuple t) { return ProductKDE::__setstate__(t); }));
}