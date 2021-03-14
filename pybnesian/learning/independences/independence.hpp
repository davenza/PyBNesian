#ifndef PYBNESIAN_LEARNING_INDEPENDENCES_INDEPENDENCE_HPP
#define PYBNESIAN_LEARNING_INDEPENDENCES_INDEPENDENCE_HPP

#include <string>
#include <vector>
#include <dataset/dataset.hpp>
#include <dataset/dynamic_dataset.hpp>
#include <util/util_types.hpp>

using dataset::DataFrame, dataset::DynamicDataFrame, dataset::DynamicVariable, dataset::DynamicAdaptator;
using util::ArcStringVector;

namespace learning::independences {

class IndependenceTest {
public:
    using int_iterator = typename std::vector<int>::const_iterator;
    using string_iterator = typename std::vector<std::string>::const_iterator;

    virtual ~IndependenceTest(){};

    virtual double pvalue(const std::string& v1, const std::string& v2) const = 0;
    virtual double pvalue(const std::string& v1, const std::string& v2, const std::string& ev) const = 0;
    virtual double pvalue(const std::string& v1, const std::string& v2, const std::vector<std::string>& ev) const = 0;

    virtual int num_variables() const = 0;
    virtual std::vector<std::string> variable_names() const = 0;
    virtual const std::string& name(int i) const = 0;
    virtual bool has_variables(const std::string& name) const = 0;
    virtual bool has_variables(const std::vector<std::string>& cols) const = 0;
};

class DynamicIndependenceTest {
public:
    virtual ~DynamicIndependenceTest() {}
    virtual const IndependenceTest& static_tests() const = 0;
    virtual const IndependenceTest& transition_tests() const = 0;

    virtual std::vector<std::string> variable_names() const = 0;
    virtual const std::string& name(int i) const = 0;
    virtual bool has_variables(const std::string& name) const = 0;
    virtual bool has_variables(const std::vector<std::string>& cols) const = 0;

    virtual int num_variables() const = 0;
    virtual int markovian_order() const = 0;
};

template <typename BaseTest>
class DynamicIndependenceTestAdaptator : public DynamicIndependenceTest, DynamicAdaptator<BaseTest> {
public:
    template <typename... Args>
    DynamicIndependenceTestAdaptator(const DynamicDataFrame& df, const Args&... args)
        : DynamicAdaptator<BaseTest>(df, args...) {}

    const IndependenceTest& static_tests() const override { return this->static_element(); }

    const IndependenceTest& transition_tests() const override { return this->transition_element(); }

    std::vector<std::string> variable_names() const override { return DynamicAdaptator<BaseTest>::variable_names(); }

    const std::string& name(int i) const override { return DynamicAdaptator<BaseTest>::name(i); }

    bool has_variables(const std::string& name) const override {
        return DynamicAdaptator<BaseTest>::has_variables(name);
    }

    bool has_variables(const std::vector<std::string>& cols) const override {
        return DynamicAdaptator<BaseTest>::has_variables(cols);
    }

    int num_variables() const override { return DynamicAdaptator<BaseTest>::num_variables(); }

    int markovian_order() const override { return DynamicAdaptator<BaseTest>::markovian_order(); }
};

}  // namespace learning::independences

#endif  // PYBNESIAN_LEARNING_INDEPENDENCES_INDEPENDENCE_HPP