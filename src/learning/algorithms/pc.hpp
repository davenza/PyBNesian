#ifndef PGM_DATASET_PC_HPP
#define PGM_DATASET_PC_HPP

#include <util/util_types.hpp>
#include <models/GaussianNetwork.hpp>

using util::ArcVector;
using models::GaussianNetwork;

namespace learning::algorithms {
    

    class PC {
        
        GaussianNetwork estimate(const DataFrame& df, ArcVector& arc_blacklist, ArcVector& arc_whitelist);

    };
}

#endif //PGM_DATASET_PC_HPP