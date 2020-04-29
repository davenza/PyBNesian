//
// Created by david on 19/4/20.
//

#ifndef PGM_DATASET_LINALG_HPP
#define PGM_DATASET_LINALG_HPP

#include <arrow/api.h>

using arrow::Array, arrow::DataType;

namespace linalg {

    double mean(std::shared_ptr<Array> ar, std::shared_ptr<DataType> dt);
    double var(std::shared_ptr<Array> ar, std::shared_ptr<DataType> dt);
}

#endif //PGM_DATASET_LINALG_HPP
