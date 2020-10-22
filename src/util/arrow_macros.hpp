#ifndef PGM_DATASET_ARROW_MACROS_HPP
#define PGM_DATASET_ARROW_MACROS_HPP

#define RAISE_STATUS_ERROR(expr)                                                        \
    {                                                                                   \
        auto __status = (expr);                                                         \
        if (!__status.ok()) {                                                           \
            throw std::runtime_error("Apache Arrow error: " + __status.ToString());     \
        }                                                                               \
    }                                                                                   

#define RAISE_RESULT_ERROR(res, expr)                                                       \
    auto __result = (expr);                                                                 \
    RAISE_STATUS_ERROR(__result.status())                                                   \
    res = std::move(__result).ValueOrDie();                                                


#endif //PGM_DATASET_ARROW_MACROS_HPP