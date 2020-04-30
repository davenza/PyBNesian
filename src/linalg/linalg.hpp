//
// Created by david on 19/4/20.
//

#ifndef PGM_DATASET_LINALG_HPP
#define PGM_DATASET_LINALG_HPP

#include <arrow/api.h>

using arrow::Array, arrow::DataType;

namespace linalg {

    double mean(Column col);
    double var(Column col);
    double var(Column col, double mean);
}

#endif //PGM_DATASET_LINALG_HPP
