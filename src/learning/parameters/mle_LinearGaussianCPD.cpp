#include <learning/parameters/mle_LinearGaussianCPD.hpp>

namespace learning::parameters {

    template<>
    typename LinearGaussianCPD::ParamsClass MLE<LinearGaussianCPD>::estimate(const DataFrame& df, 
                                                                             const std::string& variable,  
                                                                             const std::vector<std::string>& evidence) {
        auto type_id = df.same_type(variable, evidence);
        bool contains_null = df.null_count(variable, evidence) > 0;

        switch(type_id) {
            case Type::DOUBLE: {
                if (contains_null)
                    return _fit<DoubleType, true>(df, variable, evidence);
                else
                    return _fit<DoubleType, false>(df, variable, evidence);
                break;
            }
            case Type::FLOAT: {
                if (contains_null)
                    return _fit<FloatType, true>(df, variable, evidence);
                else
                    return _fit<FloatType, false>(df, variable, evidence);
                break;
            }
            default:
                throw std::invalid_argument("Wrong data type to fit the linear regression. \"double\" or \"float\" data is expected.");
        }
    }

}