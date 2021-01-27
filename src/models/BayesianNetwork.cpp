#include <models/BayesianNetwork.hpp>

namespace models {
    
    std::string BayesianNetworkType_ToString(BayesianNetworkType t) {
        switch(t) {
            case Gaussian:
                return "Gaussian";
            case Semiparametric:
                return "Semiparametric";
            case Discrete:
                return "Discrete";
            default:
                throw std::invalid_argument("Unreachable code in BayesianNetworkType.");
        }
    }

    void requires_continuous_data(const DataFrame& df) {
        auto schema = df->schema();

        if (schema->num_fields() == 0) {
            throw std::invalid_argument("Provided dataset does not contain columns.");
        }

        auto dtid = schema->field(0)->type()->id();

        if (dtid != Type::DOUBLE && dtid != Type::FLOAT) {
            throw std::invalid_argument("Continuous data (double or float) is needed to learn Gaussian networks. "
                                        "Column \"" + schema->field(0)->name() + "\" (DataType: " + schema->field(0)->type()->ToString() + ").");
        }

        for (auto i = 1; i < schema->num_fields(); ++i) {
            auto new_dtid = schema->field(i)->type()->id();
            if (dtid != new_dtid)
                throw std::invalid_argument("All the columns should have the same data type. "
                                            "Column \"" + schema->field(0)->name() + "\" (DataType: " + schema->field(0)->type()->ToString() + "). "
                                            "Column \"" + schema->field(i)->name() + "\" (DataType: " + schema->field(i)->type()->ToString() + ").");
        }
    }


    void requires_discrete_data(const DataFrame& df) {
        auto schema = df->schema();

        if (schema->num_fields() == 0) {
            throw std::invalid_argument("Provided dataset does not contain columns.");
        }

        for (auto i = 0; i < schema->num_fields(); ++i) {
            auto dtid = schema->field(i)->type()->id();
            if (dtid != Type::DICTIONARY)
                throw std::invalid_argument("Categorical data is needed to learn discrete Bayesian networks. "
                                        "Column \"" + schema->field(i)->name() + "\" (DataType: " + schema->field(i)->type()->ToString() + ").");
        }
    }



    py::object load_model(const std::string& name) {
        auto open = py::module::import("io").attr("open");
        auto file = open(name, "rb");
        auto bn = py::module::import("pickle").attr("load")(file);
        file.attr("close")();
        return bn;
    }
}