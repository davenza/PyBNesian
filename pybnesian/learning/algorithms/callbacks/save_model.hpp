#ifndef PYBNESIAN_LEARNING_ALGORITHMS_CALLBACKS_SAVE_MODEL_HPP
#define PYBNESIAN_LEARNING_ALGORITHMS_CALLBACKS_SAVE_MODEL_HPP

#include <learning/algorithms/callbacks/callback.hpp>

namespace learning::algorithms::callbacks {

class SaveModel : public Callback {
public:
    SaveModel(const std::string& folder_name) : m_folder_name(folder_name) {}

    void call(BayesianNetworkBase& model, Operator*, Score&, int num_iter) const override {
        std::stringstream file_name;
        file_name << m_folder_name << "/" << std::setfill('0') << std::setw(6) << num_iter;
        model.save(file_name.str(), false);
    }

private:
    std::string m_folder_name;
};

}  // namespace learning::algorithms::callbacks

#endif  // PYBNESIAN_LEARNING_ALGORITHMS_CALLBACKS_SAVE_MODEL_HPP