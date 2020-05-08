#ifndef PGM_DATASET_LINALG_HPP
#define PGM_DATASET_LINALG_HPP

#include <arrow/api.h>

using arrow::Array, arrow::DataType;
typedef std::shared_ptr<arrow::Array> Array_ptr;
typedef std::shared_ptr<arrow::Buffer> Buffer_ptr;

namespace linalg {

    double mean(Array_ptr col);
    double mean(Array_ptr col, Buffer_ptr bitmap);
    double var(Array_ptr col);
    double var(Array_ptr col, Buffer_ptr bitmap);
    double var(Array_ptr col, double mean);
    double var(Array_ptr col, double mean, Buffer_ptr bitmap);
    double covariance(Array_ptr col1, Array_ptr col2, double mean1, double mean2);
    double covariance(Array_ptr col1, Array_ptr col2, double mean1, double mean2, Buffer_ptr bitmap);
    double sse(Array_ptr truth, Array_ptr predicted);
    double sse(Array_ptr truth, Array_ptr predicted, Buffer_ptr bitmap);

    namespace linear_regression {
        Array_ptr fitted_values(std::vector<double> beta, std::vector<Array_ptr> columns);
    }
}

#endif //PGM_DATASET_LINALG_HPP
