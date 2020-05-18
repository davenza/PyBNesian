#include <limits>
#include <learning/operators/operators.hpp>


namespace learning::operators {


    template<typename Model, typename Score>
    void OperatorPool<Model, Score>::cache_scores(const Model& m) {
        for(auto t : operator_types) {
            t.cache_scores(m);
        }
    }

    template<typename Model, typename Score>
    void OperatorPool<Model, Score>::update_scores(const Model& m, std::unique_ptr<Operator<Model>> new_op) {
        for(auto t : operator_types) {
            t.update_scores(m, new_op);
        }
    }

    template<typename Model, typename Score>
    std::unique_ptr<Operator<Model>> OperatorPool<Model, Score>::find_max(const Model& m) {

        auto best_delta = std::numeric_limits<double>::min();
        auto best_op = nullptr;

        for (auto t : operator_types) {
            auto new_op = t.find_max(m);

            if (new_op->delta() > best_delta) {
                best_delta = new_op->delta();
                best_op = new_op;
            }
        }

        return best_op;
    }


    // template<typename Model, typename Score>
    // void ArcOperatorsType<Model, Score>::cache_scores(const Model& m) {
        

    // }


}