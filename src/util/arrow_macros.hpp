#ifndef PGM_DATASET_ARROW_MACROS_HPP
#define PGM_DATASET_ARROW_MACROS_HPP

#define RAISE_STATUS_ERROR(expr)                                                        \
    {                                                                                   \
        auto __status = (expr);                                                         \
        if (!__status.ok()) {                                                           \
            throw std::runtime_error("Apache Arrow error: " + __status.ToString());     \
        }                                                                               \
    }                                                                                   

#endif //PGM_DATASET_ARROW_MACROS_HPP