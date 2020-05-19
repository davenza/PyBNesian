#include <limits>
#include <learning/operators/operators.hpp>


namespace learning::operators {


    template<typename Model, typename Score>
    void ArcOperatorsType<Model, Score>::cache_scores(const DataFrame& df, const Model& m) {
        
        using node_descriptor = Model::node_descriptor;

        for (auto i = 0; i < df.num_columns(); ++i) {
            auto col = df->column(i);


        }

    }
}