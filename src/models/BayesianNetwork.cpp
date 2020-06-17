#include <models/BayesianNetwork.hpp>

namespace models {
    
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
}