#ifndef PYBNESIAN_FACTORS_ARGUMENTS_HPP
#define PYBNESIAN_FACTORS_ARGUMENTS_HPP

#include <pybind11/pybind11.h>
#include <factors/factors.hpp>
#include <util/hash_utils.hpp>
#include <util/util_types.hpp>

using util::FactorTypeHash, util::FactorTypeEqualTo, util::PairNameType, util::NameFactorTypeHash,
    util::NameFactorTypeEqualTo;

namespace py = pybind11;

namespace factors {

class Args {
public:
    Args(py::args args) : m_args(args) {}

    py::args args() { return m_args; }

private:
    py::args m_args;
};

class Kwargs {
public:
    Kwargs(py::kwargs kwargs) : m_kwargs(kwargs) {}

    py::kwargs kwargs() { return m_kwargs; }

private:
    py::kwargs m_kwargs;
};

class Arguments {
public:
    Arguments() = default;

    Arguments(py::dict dict_arguments) {
        for (const auto arg_params : dict_arguments) {
            try {
                auto str = py::cast<std::string>(arg_params.first);
                auto args = process_args(arg_params.second);
                m_name_args.insert({str, args});
                continue;
            } catch (py::cast_error& ce) {
            }

            try {
                auto ft = py::cast<std::shared_ptr<FactorType>>(arg_params.first);
                auto args = process_args(arg_params.second);
                m_type_args.insert({ft, args});
                continue;
            } catch (py::cast_error& ce) {
            }

            if (py::isinstance<py::tuple>(arg_params.first)) {
                auto tuple = py::cast<py::tuple>(arg_params.first);
                if (tuple.size() == 2) {
                    try {
                        auto str = py::cast<std::string>(tuple[0]);
                        auto ft = py::cast<std::shared_ptr<FactorType>>(tuple[1]);
                        auto args = process_args(arg_params.second);
                        m_name_type_args.insert({std::make_pair(str, ft), args});
                        continue;
                    } catch (py::cast_error& ce) {
                    }
                }
            }

            throw std::invalid_argument("Key value is not of type str, FactorType or 2-tuple (str, FactorType).");
        }
    }

    std::pair<py::args, py::kwargs> args(const std::string& name, const std::shared_ptr<FactorType>& ft) const {
        auto str_ft_params = m_name_type_args.find(std::make_pair(name, ft));
        if (str_ft_params != m_name_type_args.end()) {
            return str_ft_params->second;
        }

        auto str_params = m_name_args.find(name);
        if (str_params != m_name_args.end()) {
            return str_params->second;
        }

        auto ft_params = m_type_args.find(ft);
        if (ft_params != m_type_args.end()) {
            return ft_params->second;
        }

        return std::make_pair(py::args{}, py::kwargs{});
    }

private:
    std::pair<py::args, py::kwargs> process_args(py::handle o) const {
        if (py::isinstance<py::tuple>(o)) {
            auto tuple = py::cast<py::tuple>(o);

            // Test 2-tuple (Args(...), Kwargs(...)).
            if (tuple.size() == 2) {
                try {
                    auto CArgs = py::cast<Args>(tuple[0]);
                    auto CKwargs = py::cast<Kwargs>(tuple[1]);
                    return std::make_pair(CArgs.args(), CKwargs.kwargs());
                } catch (py::cast_error& ce) {
                }
            }

            // This is a tuple of Args.
            return std::make_pair(py::cast<py::args>(o), py::kwargs{});
        }

        // This is an py::kwargs object
        if (py::isinstance<py::kwargs>(o)) {
            return std::make_pair(py::args{}, py::cast<py::kwargs>(o));
        }

        // This is an Args(...) object.
        try {
            auto CArgs = py::cast<Args>(o);
            return std::make_pair(CArgs.args(), py::kwargs{});
        } catch (py::cast_error& ce) {
        }

        // This is an Kwargs(...) object
        try {
            auto CKwargs = py::cast<Kwargs>(o);
            return std::make_pair(py::args{}, CKwargs.kwargs());
        } catch (py::cast_error& ce) {
        }

        throw std::invalid_argument(
            "The provided arguments must be a 2-tuple (Args(...), Kwargs(...)),"
            " an Args(...) (or tuple) or a Kwargs(...) (or dict).");
    }

    std::unordered_map<std::string, std::pair<py::args, py::kwargs>> m_name_args;
    std::unordered_map<std::shared_ptr<FactorType>, std::pair<py::args, py::kwargs>, FactorTypeHash, FactorTypeEqualTo>
        m_type_args;
    std::unordered_map<PairNameType, std::pair<py::args, py::kwargs>, NameFactorTypeHash, NameFactorTypeEqualTo>
        m_name_type_args;
};

}  // namespace factors

#endif  // PYBNESIAN_FACTORS_ARGUMENTS_HPP