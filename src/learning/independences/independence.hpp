#ifndef PYBNESIAN_LEARNING_INDEPENDENCES_INDEPENDENCE_HPP
#define PYBNESIAN_LEARNING_INDEPENDENCES_INDEPENDENCE_HPP

#include <string>
#include <vector>
#include <dataset/dataset.hpp>
#include <dataset/dynamic_dataset.hpp>

using dataset::DataFrame, dataset::DynamicDataFrame, dataset::DynamicVariable;

namespace learning::independences {
    class IndependenceTest {
    public:
        using int_iterator = typename std::vector<int>::const_iterator;
        using string_iterator = typename std::vector<std::string>::const_iterator;

        virtual ~IndependenceTest() {};
        
        virtual double pvalue(int v1, int v2) const = 0;
        virtual double pvalue(const std::string& v1, const std::string& v2) const = 0;

        virtual double pvalue(int v1, int v2, int cond) const = 0;
        virtual double pvalue(const std::string& v1, const std::string& v2, const std::string& cond) const = 0;

        virtual double pvalue(int v1, int v2, 
                              const int_iterator evidence_begin, 
                              const int_iterator evidence_end) const = 0;

        virtual double pvalue(const std::string& v1, const std::string& v2, 
                              const string_iterator evidence_begin, 
                              const string_iterator evidence_end) const = 0;
        
        virtual std::vector<std::string> column_names() const = 0;
        virtual const std::string& name(int i) const = 0;
        virtual int num_columns() const = 0;
    };

    class DynamicIndependenceTest {
        using int_iterator = typename std::vector<DynamicVariable<int>>::const_iterator;
        using string_iterator = typename std::vector<DynamicVariable<std::string>>::const_iterator;
        
        virtual ~DynamicIndependenceTest() {}

        virtual double pvalue(DynamicVariable<int> v1,
                              DynamicVariable<int> v2) const = 0;
        virtual double pvalue(DynamicVariable<std::string> v1,
                              DynamicVariable<std::string> v2) const = 0;
        
        virtual double pvalue(DynamicVariable<int> v1,
                              DynamicVariable<int> v2,
                              DynamicVariable<int> cond) const = 0;
        virtual double pvalue(DynamicVariable<std::string> v1,
                              DynamicVariable<std::string> v2,
                              DynamicVariable<std::string> cond) const = 0;
        
        virtual double pvalue(DynamicVariable<int> v1,
                              DynamicVariable<int> v2,
                              const int_iterator evidence_begin,
                              const int_iterator evidence_end) const = 0;
        virtual double pvalue(DynamicVariable<std::string> v1,
                              DynamicVariable<std::string> v2,
                              const string_iterator evidence_begin,
                              const string_iterator evidence_end) const = 0;

        virtual int num_columns() const = 0;
        virtual int num_temporal_slices() const = 0;
    };

    template<typename DataFrameType>
    struct independence_traits;

    template<>
    struct independence_traits<DataFrame> {
        using independence_base = IndependenceTest;
    };

    template<>
    struct independence_traits<DynamicDataFrame> {
        using independence_base = DynamicIndependenceTest;
    };

}

#endif //PYBNESIAN_LEARNING_INDEPENDENCES_INDEPENDENCE_HPP