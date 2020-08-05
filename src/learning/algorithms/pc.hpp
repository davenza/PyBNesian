#ifndef PGM_DATASET_PC_HPP
#define PGM_DATASET_PC_HPP

#include <util/util_types.hpp>
#include <models/GaussianNetwork.hpp>
#include <learning/independences/independence.hpp>

using util::ArcVector;
using models::GaussianNetwork;
using learning::independences::IndependenceTest;

namespace learning::algorithms {
    

    class PC {
    public:
        void estimate(const DataFrame& df, 
                        ArcVector& arc_blacklist, 
                        ArcVector& arc_whitelist, 
                        const IndependenceTest& test,
                        double alpha);

    };
}

#endif //PGM_DATASET_PC_HPP