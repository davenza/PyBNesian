#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <factors/factors.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/continuous/CKDE.hpp>
#include <factors/discrete/DiscreteFactor.hpp>
#include <factors/factors.hpp>
#include <util/util_types.hpp>

namespace py = pybind11;

using factors::Factor, factors::continuous::LinearGaussianCPD, factors::continuous::KDE, factors::continuous::CKDE;
using factors::FactorType, factors::continuous::LinearGaussianCPDType, factors::continuous::CKDEType,
    factors::discrete::DiscreteFactorType;
using factors::discrete::DiscreteFactor;
using util::random_seed_arg;

class PyFactorType : public FactorType {
public:
    using FactorType::FactorType;
    PyFactorType(const PyFactorType&) = delete;
    void operator=(const PyFactorType&) = delete;

    PyFactorType() { m_hash = reinterpret_cast<std::uintptr_t>(nullptr); }

    bool is_python_derived() const override { return true; }

    std::shared_ptr<Factor> new_factor(const std::string& variable,
                                       const std::vector<std::string>& parents) const override {
        pybind11::gil_scoped_acquire gil;
        pybind11::function override = pybind11::get_override(static_cast<const FactorType*>(this), "new_factor");

        if (override) {
            auto o = override(variable, parents);
            auto keep_python_state_alive = std::make_shared<py::object>(o);
            auto ptr = o.cast<Factor*>();

            return std::shared_ptr<Factor>(keep_python_state_alive, ptr);
        }

        py::pybind11_fail("Tried to call pure virtual function \"FactorType::new_factor\"");
    }

    std::shared_ptr<FactorType> opposite_semiparametric() const override {
        pybind11::gil_scoped_acquire gil;
        pybind11::function override =
            pybind11::get_override(static_cast<const FactorType*>(this), "opposite_semiparametric");

        if (override) {
            auto o = override();
            auto keep_python_state_alive = std::make_shared<py::object>(o);
            auto ptr = o.cast<FactorType*>();

            return std::shared_ptr<FactorType>(keep_python_state_alive, ptr);
        }

        py::pybind11_fail("Tried to call pure virtual function \"FactorType::opposite_semiparametric\"");
    }

    std::string ToString() const override { PYBIND11_OVERRIDE_PURE(std::string, FactorType, ToString, ); }

    size_t hash() const override {
        if (m_hash == reinterpret_cast<std::uintptr_t>(nullptr)) {
            py::object o = py::cast(this);
            py::handle ttype = o.get_type();
            // Get the pointer of the Python derived type class.
            // !!!! We have to do this here because in the constructor,
            // "this" is just a FactorType instead of the derived Python class !!!!!!!!!!!!!!!
            m_hash = reinterpret_cast<std::uintptr_t>(ttype.ptr());
        }

        return m_hash;
    }

    py::tuple __getstate__() const override {
        py::gil_scoped_acquire gil;
        py::function override = py::get_override(static_cast<const FactorType*>(this), "__getstate_extra__");
        if (override) {
            return py::make_tuple(true, override());
        } else {
            return py::make_tuple(false, py::make_tuple());
        }
    }

    static void __setstate__(py::object& self, py::tuple& t) {
        // Call trampoline constructor
        py::gil_scoped_acquire gil;
        auto pyfactortype = py::type::of<FactorType>();
        pyfactortype.attr("__init__")(self);

        auto ptr = self.cast<const FactorType*>();

        auto extra_info = t[0].cast<bool>();
        if (extra_info) {
            py::function override = py::get_override(ptr, "__setstate_extra__");
            if (override) {
                override(t[1]);
            } else {
                py::pybind11_fail("Tried to call function \"FactorType::__setstate_extra__\"");
            }
        }
    }
};

class PyFactor : public Factor {
public:
    // This class is needed to implement a constructor for Factor that accepts a FactorType.
    class CustomFactorType : public FactorType {
    public:
        CustomFactorType(const Factor* pfactor) : m_pfactor(pfactor), m_pyderived(false), m_pyinstance(), m_name() {
            pybind11::gil_scoped_acquire gil;
            // Get Python derived FactorType class name.
            py::object o = py::cast(m_pfactor);
            py::handle ttype = o.get_type();
            auto class_name = ttype.attr("__name__").cast<std::string>();
            m_name = class_name + "Type";
            // Here, we use the pointer of the Factor class as hash.
            m_hash = reinterpret_cast<std::uintptr_t>(ttype.ptr());
        }

        CustomFactorType(const Factor* pfactor, py::object& factor_type)
            : m_pfactor(pfactor), m_pyderived(true), m_pyinstance(factor_type), m_name() {
            m_hash = reinterpret_cast<std::uintptr_t>(factor_type.get_type().ptr());
        }

        std::shared_ptr<Factor> new_factor(const std::string& variable,
                                           const std::vector<std::string>& parents) const override {
            py::gil_scoped_acquire gil;

            py::function override = py::get_override(m_pfactor, "new_factor");
            if (override) {
                auto o = override(variable, parents);
                auto keep_python_state_alive = std::make_shared<py::object>(o);
                auto ptr = o.cast<Factor*>();

                return std::shared_ptr<Factor>(keep_python_state_alive, ptr);
            }

            py::pybind11_fail("Tried to call pure virtual function \"Factor::new_factor\"");
        }

        std::shared_ptr<FactorType> opposite_semiparametric() const override { return nullptr; }

        std::string ToString() const override {
            if (m_pyderived) {
                py::gil_scoped_acquire gil;

                py::function override = py::get_override(m_pyinstance.cast<const FactorType*>(), "ToString");
                if (override) {
                    auto o = override();
                    return o.cast<std::string>();
                }

                py::pybind11_fail("Tried to call pure virtual function \"FactorType::ToString\"");
            } else {
                return m_name;
            }
        }

        py::tuple __getstate__() const override {
            if (m_pyderived) {
                auto bytes = py::module::import("pickle").attr("dumps")(m_pyinstance);
                return py::make_tuple(true, bytes);
            } else {
                return py::make_tuple(false);
            }
        }

        static std::shared_ptr<CustomFactorType> __setstate__(const Factor* pyf, py::tuple& t) {
            auto pyderived = t[0].cast<bool>();
            if (pyderived) {
                auto bytes = py::module::import("pickle").attr("loads")(t[1]);
                return std::make_shared<CustomFactorType>(pyf, bytes);
            } else {
                return nullptr;
            }
        }

        bool is_pyderived() { return m_pyderived; }

        py::object& pyinstance() { return m_pyinstance; }

    private:
        const Factor* m_pfactor;
        bool m_pyderived;
        py::object m_pyinstance;
        std::string m_name;
    };

    PyFactor(const std::string& variable, const std::vector<std::string>& parents, bool create_factor_type = true)
        : Factor(variable, parents), m_type(nullptr), m_create_factor_type(create_factor_type) {}

    bool is_factortype_subclass(py::object& factor_type_class) {
        auto factor_type = py::type::of<FactorType>();
        int subclass = PyObject_IsSubclass(factor_type_class.ptr(), factor_type.ptr());
        return subclass == 1;
    }

    PyFactor(const std::string& variable, const std::vector<std::string>& parents, py::object& factor_type_class)
        : Factor(variable, parents), m_type(nullptr), m_create_factor_type(false) {
        if (!py::isinstance<py::type>(factor_type_class) || !is_factortype_subclass(factor_type_class)) {
            throw std::invalid_argument("\"factor_type\" argument must be a class type that inherits FactorType");
        }

        auto new_factor_type = factor_type_class();
        m_type = std::make_shared<CustomFactorType>(this, new_factor_type);
    }

    std::shared_ptr<FactorType> type() const override {
        if (!m_type) {
            if (m_create_factor_type) {
                // Generate custom factor type if it was not defined in the constructor.
                // !!!! We have to do this here because in the constructor
                // "this" is just a FactorType instead of the derived Python class !!!!!!!!!!!!!!!
                m_type = std::make_shared<CustomFactorType>(this);
            } else {
                py::gil_scoped_acquire gil;

                pybind11::function override = pybind11::get_override(static_cast<const Factor*>(this), "type");
                if (override) {
                    auto o = override();
                    m_type = o.cast<std::shared_ptr<FactorType>>();
                    m_type = FactorType::keep_python_alive(m_type);
                    return m_type;
                }

                py::pybind11_fail("Tried to call pure virtual function \"Factor::type\"");
            }
        }

        return m_type;
    }

    FactorType& type_ref() const override { return *type(); }

    std::shared_ptr<arrow::DataType> data_type() const override {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<arrow::DataType>, Factor, data_type, );
    }

    bool fitted() const override { PYBIND11_OVERRIDE_PURE_NAME(bool, Factor, "is_fitted", fitted, ); }

    void fit(const DataFrame& df) override { PYBIND11_OVERRIDE_PURE(void, Factor, fit, df); }

    VectorXd logl(const DataFrame& df) const override { PYBIND11_OVERRIDE_PURE(VectorXd, Factor, logl, df); }

    double slogl(const DataFrame& df) const override { PYBIND11_OVERRIDE_PURE(double, Factor, slogl, df); }

    std::string ToString() const override { PYBIND11_OVERRIDE_PURE(std::string, Factor, ToString, ); }

    Array_ptr sample(int n,
                     const DataFrame& evidence_values,
                     unsigned int seed = std::random_device{}()) const override {
        PYBIND11_OVERRIDE_PURE(Array_ptr, Factor, sample, n, evidence_values, seed);
    }

    py::tuple __getstate__() const override {
        auto& t = type_ref();

        py::gil_scoped_acquire gil;
        pybind11::function override = pybind11::get_override(static_cast<const Factor*>(this), "__getstate_extra__");
        if (override) {
            auto o = override();
            return py::make_tuple(
                variable(), evidence(), m_create_factor_type, t.__getstate__(), true, py::make_tuple(o));
        } else {
            return py::make_tuple(
                variable(), evidence(), m_create_factor_type, t.__getstate__(), false, py::make_tuple());
        }
    }

    static void __setstate__(py::object& self, py::tuple& t) {
        auto v = t[0].cast<std::string>();
        auto p = t[1].cast<std::vector<std::string>>();
        auto create_factor = t[2].cast<bool>();
        auto type_tuple = t[3].cast<py::tuple>();

        py::gil_scoped_acquire gil;

        bool has_ctype = type_tuple[0].cast<bool>();

        auto pyfactor_class = py::type::of<Factor>();
        if (has_ctype) {
            pyfactor_class.attr("__init__")(self, v, p, false);
            auto self_cpp = self.cast<PyFactor*>();
            self_cpp->m_type = CustomFactorType::__setstate__(self_cpp, type_tuple);
        } else {
            pyfactor_class.attr("__init__")(self, v, p, create_factor);
        }

        bool is_extra = t[4].cast<bool>();
        if (is_extra) {
            pybind11::function override = pybind11::get_override(self.cast<const Factor*>(), "__setstate_extra__");
            if (override) {
                auto extra_info = t[5].cast<py::tuple>();
                override(extra_info[0]);
            } else {
                py::pybind11_fail("Tried to call \"Factor::__setstate_extra__\"");
            }
        }
    }

private:
    mutable std::shared_ptr<FactorType> m_type;
    bool m_create_factor_type;
};

void pybindings_factors(py::module& root) {
    auto factors = root.def_submodule("factors", "Factors submodule.");

    py::class_<FactorType, PyFactorType, std::shared_ptr<FactorType>>(factors, "FactorType")
        .def(py::init<>())
        .def("new_factor", &FactorType::new_factor)
        .def("opposite_semiparametric", &FactorType::opposite_semiparametric)
        .def("ToString", &FactorType::ToString)
        .def("hash", &FactorType::hash)
        // The equality operator do not compile in GCC, so it is implemented with lambdas:
        // https://github.com/pybind/pybind11/issues/1487
        .def(
            "__eq__", [](const FactorType& self, const FactorType& other) { return self == other; }, py::is_operator())
        .def(
            "__ne__", [](const FactorType& self, const FactorType& other) { return self != other; }, py::is_operator())
        // .def(py::self == py::self)
        // .def(py::self != py::self)
        .def("__getstate__", [](const FactorType& self) { return self.__getstate__(); })
        // Setstate for pyderived type
        .def("__setstate__", [](py::object& self, py::tuple& t) { PyFactorType::__setstate__(self, t); })
        .def("__repr__", [](const FactorType& self) { return self.ToString(); })
        .def("__str__", [](const FactorType& self) { return self.ToString(); });

    py::class_<Factor, PyFactor, std::shared_ptr<Factor>>(factors, "Factor")
        .def(py::init<const std::string&, const std::vector<std::string>&, bool>(),
             py::arg("variable"),
             py::arg("parents"),
             py::arg("create_factor_type") = true)
        .def(py::init<const std::string&, const std::vector<std::string>&, py::object&>(),
             py::arg("variable"),
             py::arg("parents"),
             py::arg("factor_type_class"))
        .def_property_readonly("variable", &Factor::variable)
        .def_property_readonly("evidence", &Factor::evidence)
        .def_property_readonly("fitted", &Factor::fitted)
        .def("type", &Factor::type)
        .def("data_type", &Factor::data_type)
        .def("fit", &Factor::fit)
        .def("logl", &Factor::logl, py::return_value_policy::take_ownership)
        .def("slogl", &Factor::slogl)
        .def(
            "sample",
            [](const Factor& self,
               int n,
               std::optional<const DataFrame> evidence_values,
               std::optional<unsigned int> seed) {
                if (evidence_values)
                    return self.sample(n, *evidence_values, random_seed_arg(seed));
                else
                    return self.sample(n, DataFrame(), random_seed_arg(seed));
            },
            py::arg("n"),
            py::arg("evidence_values") = std::nullopt,
            py::arg("seed") = std::nullopt)
        .def("save", &Factor::save)
        .def("ToString", &Factor::ToString)
        .def("__str__", &Factor::ToString)
        .def("__repr__", &Factor::ToString)
        .def("__getstate__", [](const Factor& self) { return self.__getstate__(); })
        .def("__setstate__", [](py::object& self, py::tuple& t) { PyFactor::__setstate__(self, t); });

    auto continuous = factors.def_submodule("continuous", "Continuous factors submodule.");

    py::class_<LinearGaussianCPDType, FactorType, std::shared_ptr<LinearGaussianCPDType>>(continuous,
                                                                                          "LinearGaussianCPDType")
        .def(py::init(&LinearGaussianCPDType::get))
        .def(py::pickle([](const LinearGaussianCPDType& self) { return self.__getstate__(); },
                        [](py::tuple&) { return LinearGaussianCPDType::get(); }));

    py::class_<LinearGaussianCPD, Factor, std::shared_ptr<LinearGaussianCPD>>(continuous, "LinearGaussianCPD")
        .def(py::init<const std::string, const std::vector<std::string>>())
        .def(py::init<const std::string, const std::vector<std::string>, const std::vector<double>, double>())
        .def_property("beta", &LinearGaussianCPD::beta, &LinearGaussianCPD::set_beta)
        .def_property("variance", &LinearGaussianCPD::variance, &LinearGaussianCPD::set_variance)
        .def("cdf", &LinearGaussianCPD::cdf, py::return_value_policy::take_ownership)
        .def(py::pickle([](const LinearGaussianCPD& self) { return self.__getstate__(); },
                        [](py::tuple t) { return LinearGaussianCPD::__setstate__(t); }));

    py::class_<KDE>(continuous, "KDE")
        .def(py::init<std::vector<std::string>>())
        .def_property_readonly("variables", &KDE::variables)
        .def_property_readonly("N", &KDE::num_instances)
        .def_property_readonly("d", &KDE::num_variables)
        .def_property("bandwidth", &KDE::bandwidth, &KDE::setBandwidth)
        .def_property_readonly("dataset", &KDE::training_data)
        .def_property_readonly("fitted", &KDE::fitted)
        .def("data_type", &KDE::data_type)
        .def("fit", (void (KDE::*)(const DataFrame&)) & KDE::fit)
        .def("logl", &KDE::logl, py::return_value_policy::take_ownership)
        .def("slogl", &KDE::slogl)
        .def("save", &KDE::save)
        .def(py::pickle([](const KDE& self) { return self.__getstate__(); },
                        [](py::tuple t) { return KDE::__setstate__(t); }));

    py::class_<CKDEType, FactorType, std::shared_ptr<CKDEType>>(continuous, "CKDEType")
        .def(py::init(&CKDEType::get))
        .def(py::pickle([](const CKDEType& self) { return self.__getstate__(); },
                        [](py::tuple&) { return CKDEType::get(); }));

    py::class_<CKDE, Factor, std::shared_ptr<CKDE>>(continuous, "CKDE")
        .def(py::init<const std::string, const std::vector<std::string>>())
        // .def_property_readonly("node_type", &CKDE::node_type)
        .def_property_readonly("N", &CKDE::num_instances)
        .def_property_readonly("kde_joint", &CKDE::kde_joint, py::return_value_policy::reference_internal)
        .def_property_readonly("kde_marg", &CKDE::kde_marg, py::return_value_policy::reference_internal)
        .def("cdf", &CKDE::cdf, py::return_value_policy::take_ownership)
        .def(py::pickle([](const CKDE& self) { return self.__getstate__(); },
                        [](py::tuple t) { return CKDE::__setstate__(t); }));

    auto discrete = factors.def_submodule("discrete", "Discrete factors submodule.");

    py::class_<DiscreteFactorType, FactorType, std::shared_ptr<DiscreteFactorType>>(discrete, "DiscreteFactorType")
        .def(py::init(&DiscreteFactorType::get))
        .def(py::pickle([](const DiscreteFactorType& self) { return self.__getstate__(); },
                        [](py::tuple&) { return DiscreteFactorType::get(); }));

    py::class_<DiscreteFactor, Factor, std::shared_ptr<DiscreteFactor>>(discrete, "DiscreteFactor")
        .def(py::init<std::string, std::vector<std::string>>())
        .def(py::pickle([](const DiscreteFactor& self) { return self.__getstate__(); },
                        [](py::tuple t) { return DiscreteFactor::__setstate__(t); }));
}