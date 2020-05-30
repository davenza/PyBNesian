#ifndef PGM_DATASET_CKDE_HPP
#define PGM_DATASET_CKDE_HPP

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace factors::continuous {
    


    class CKDE {
    public:
        CKDE(const std::string variable, const std::vector<std::string> evidence) : m_variable(variable), m_evidence(evidence) {}

        void fit(py::handle pyobject);
    private:
        std::string m_variable;
        std::vector<std::string> m_evidence;
    };
}

#endif //PGM_DATASET_CKDE_HPP