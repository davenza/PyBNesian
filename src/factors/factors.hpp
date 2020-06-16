#ifndef PGM_DATASET_FACTORS_HPP
#define PGM_DATASET_FACTORS_HPP

namespace factors {

    template<typename Derived>
    class Factor {
    public:
        Factor(const std::string variable, const std::vector<std::string> evidence);

    private:
    };
}


#endif //PGM_DATASET_BAYESIANNETWORK_HPP