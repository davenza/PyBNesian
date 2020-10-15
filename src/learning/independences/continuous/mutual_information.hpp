#ifndef PGM_DATASET_MUTUAL_INFORMATION_HPP
#define PGM_DATASET_MUTUAL_INFORMATION_HPP

#include <dataset/dataset.hpp>
#include <learning/independences/continuous/kdtree.hpp>

using dataset::DataFrame;


namespace learning::independences {


    class KMutualInformation {
    public:
        KMutualInformation(DataFrame df, int k) : m_df(df), m_kdtree(df), m_k(k) {}

    private:
        DataFrame m_df;
        KDTree m_kdtree;
        int m_k;
    };


}

#endif //PGM_DATASET_MUTUAL_INFORMATION_HPP