#include <dataset/dataset.hpp>


namespace dataset {


    class CrossValidation {
    public:

        CrossValidation(const &DataFrame df, int k) : df(df), k(k), indices(df->num_rows()) {
            std::iota(indices.begin(), indices.end(), 0);

            auto rng = std::default_random_engine {};
            std::shuffle(indices.begin(), indices.end(), rng);
        }

        CrossValidation(const &DataFrame df, int k, int seed) : df(df), k(k), indices(df->num_rows()) {
            std::iota(indices.begin(), indices.end(), 0);

            auto rng = std::default_random_engine {seed};
            std::shuffle(indices.begin(), indices.end(), rng);
        }

    private:
        const DataFrame& df;
        int k;
        std::vector<int> indices;
    };
}