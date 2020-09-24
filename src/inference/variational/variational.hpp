#ifndef PGM_DATASET_VARIATIONAL_HPP
#define PGM_DATASET_VARIATIONAL_HPP

#include <models/BayesianNetwork.hpp>

using models::BayesianNetworkBase;

namespace inference {


    class VariationalBayes {
        VariationalBayes(BayesianNetworkBase& bn) : m_bn(bn) {}


        


        template<typename QueryType, typename EvidenceType>
        void query(QueryType query_begin, QueryType query_end, EvidenceType ev_begin, EvidenceType ev_end) {

        }


    private:
        BayesianNetworkBase& m_bn;
    };
}


#endif //PGM_DATASET_VARIATIONAL_HPP