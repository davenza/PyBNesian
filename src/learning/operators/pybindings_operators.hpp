#ifndef PGM_DATASET_PYBINDINGS_OPERATORS_HPP
#define PGM_DATASET_PYBINDINGS_OPERATORS_HPP

#include <pybind11/pybind11.h>
#include <models/BayesianNetwork.hpp>


namespace py = pybind11;

using models::BayesianNetworkType;

namespace learning::operators {

    template<typename Model, typename Score>
    ArcOperatorSet<Score> ArcOperatorSet_constructor(Model& model, const Score score, ArcVector& whitelist, ArcVector& blacklist,
                       int max_indegree) {
        return ArcOperatorSet(model, score, whitelist, blacklist, max_indegree);
    }
    

    template<typename Model, typename Score>
    ChangeNodeTypeSet<Score> ChangeNodeTypeSet_constructor(Model& model, const Score score, FactorTypeVector& type_whitelist) {
        return ChangeNodeTypeSet(model, score, type_whitelist);
    }

    template<typename Model, typename Score>
    OperatorPool<Score> OperatorPool_constructor(Model& model, const Score score, 
                                            std::vector<std::shared_ptr<OperatorSet>>& op_sets) {
        return OperatorPool(model, score, op_sets);
    }


    // template<typename Model, typename Score>
    // ArcOperatorSet<Model, Score> arcoperatorset_wrapper_constructor(Model& model,
    //                                              const Score score,
    //                                              ArcVector& blacklist,
    //                                              ArcVector& whitelist,
    //                                              int max_indegree) {
    //     return ArcOperatorSet(model, score, blacklist, whitelist, max_indegree);
    // }
}

#endif //PGM_DATASET_PYBINDINGS_OPERATORS_HPP