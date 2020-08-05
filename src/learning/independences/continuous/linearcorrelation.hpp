#ifndef PGM_DATASET_LINEARCORRELATION_HPP
#define PGM_DATASET_LINEARCORRELATION_HPP

#include <dataset/dataset.hpp>
#include <learning/independences/independence.hpp>

using dataset::DataFrame;
using learning::independences::IndependenceTest;

namespace learning::independences::continuous {

    class LinearCorrelation : public IndependenceTest {
    public:
        LinearCorrelation(const DataFrame& df) : m_df(df) {}

        double pvalue(int v1, int v2) const override;
        double pvalue(const std::string& v1, const std::string& v2) const override {}
        double pvalue(int v1, int v2, int cond) const override {}
        double pvalue(const std::string& v1, const std::string& v2, const std::string& cond) const override {}
        double pvalue(int v1, int v2, 
                        const typename std::vector<int>::const_iterator evidence_begin, 
                        const typename std::vector<int>::const_iterator evidence_end) const override {
            
        }

         double pvalue(const std::string& v1, const std::string& v2, 
                            const typename std::vector<std::string>::const_iterator evidence_begin, 
                            const typename std::vector<std::string>::const_iterator evidence_end) const override {

        }

    private:
        const DataFrame m_df;
    };
}


#endif //PGM_DATASET_LINEARCORRELATION_HPP