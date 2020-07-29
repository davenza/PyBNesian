#ifndef PGM_DATASET_SCORES_HPP
#define PGM_DATASET_SCORES_HPP

#include <models/GaussianNetwork.hpp>
#include <models/SemiparametricBN.hpp>

using models::GaussianNetwork, models::SemiparametricBN;

namespace learning::scores {

    class ScoreType
    {
    public:
        enum Value : uint8_t
        {
            BIC,
            PREDICTIVE_LIKELIHOOD
        };

        struct Hash
        {
            inline std::size_t operator ()(ScoreType const score_type) const
            {
                return static_cast<std::size_t>(score_type.value);
            }
        };

        using HashType = Hash;

        ScoreType() = default;
        constexpr ScoreType(Value opset_type) : value(opset_type) { }

        operator Value() const { return value; }  
        explicit operator bool() = delete;

        constexpr bool operator==(ScoreType a) const { return value == a.value; }
        constexpr bool operator==(Value v) const { return value == v; }
        constexpr bool operator!=(ScoreType a) const { return value != a.value; }
        constexpr bool operator!=(Value v) const { return value != v; }

        std::string ToString() const { 
            switch(value) {
                case Value::BIC:
                    return "bic";
                case Value::PREDICTIVE_LIKELIHOOD:
                    return "predic-l";
                default:
                    throw std::invalid_argument("Unreachable code in ScoreType.");
            }
        }

    private:
        Value value;
    };

    template<typename... Models>
    class ScoreInterface {};

    template<typename Model>
    class ScoreInterface<Model> {
    public:
        virtual double score(const Model& m) const {
            throw std::invalid_argument("Score::score() not implemented for model " + m.type().ToString());
        }
        virtual double local_score(const Model& m, const int) const {
            throw std::invalid_argument("Score::local_score() not implemented for model " + m.type().ToString());
        }
        virtual double local_score(const Model& m, const std::string&) const {
            throw std::invalid_argument("Score::local_score() not implemented for  model " + m.type().ToString());
        }
        virtual double local_score(const Model& m, 
                                                    const int, 
                                                    const typename std::vector<int>::const_iterator, 
                                                    const typename std::vector<int>::const_iterator) const {
            throw std::invalid_argument("Score::local_score() not implemented for model " + m.type().ToString());
        }
        virtual double local_score(const Model& m, 
                                                    const std::string&, 
                                                    const typename std::vector<std::string>::const_iterator, 
                                                    const typename std::vector<std::string>::const_iterator) const {
            throw std::invalid_argument("Score::local_score() not implemented for model " + m.type().ToString());
        }
    };

    template<typename Model, typename... Models>
    class ScoreInterface<Model, Models...> : public ScoreInterface<Models...> {
    public:
        using Base = ScoreInterface<Models...>;
        using Base::score;
        using Base::local_score;
        virtual double score(const Model& m) const {
            throw std::invalid_argument("Score::score() not implemented for model " + m.type().ToString());
        }
        virtual double local_score(const Model& m, const int) const {
            throw std::invalid_argument("Score::local_score() not implemented for model " + m.type().ToString());
        }
        virtual double local_score(const Model& m, const std::string&) const {
            throw std::invalid_argument("Score::local_score() not implemented for model " + m.type().ToString());
        }
        virtual double local_score(const Model& m, 
                            const int, 
                            const typename std::vector<int>::const_iterator, 
                            const typename std::vector<int>::const_iterator) const {
            throw std::invalid_argument("Score::local_score() not implemented for model " + m.type().ToString());
        }
        virtual double local_score(const Model& m, 
                            const std::string&, 
                            const typename std::vector<std::string>::const_iterator, 
                            const typename std::vector<std::string>::const_iterator) const {
            throw std::invalid_argument("Score::local_score() not implemented for model " + m.type().ToString());
        }
    };

    class Score : public ScoreInterface<GaussianNetwork, SemiparametricBN> {
    public:
        using Base = ScoreInterface<GaussianNetwork, SemiparametricBN>;
        using Base::score;
        using Base::local_score;
        virtual std::string ToString() const = 0;
        virtual bool is_decomposable() const = 0;
        virtual ScoreType type() const = 0;

        virtual double local_score(FactorType, int, 
                                   const typename std::vector<int>::const_iterator, 
                                   const typename std::vector<int>::const_iterator) const {
            throw std::invalid_argument("Score::local_score() not implemented for score " + this->type().ToString());
        }

        virtual double local_score(FactorType, const std::string&, 
                                   const typename std::vector<std::string>::const_iterator, 
                                   const typename std::vector<std::string>::const_iterator) const {
            throw std::invalid_argument("Score::local_score() not implemented for score " + this->type().ToString());
        }
    };


    template<typename Derived, typename... Models>
    class ScoreImpl {};

    template<typename Derived, typename Model>
    class ScoreImpl<Derived, Model> : public Score {
    public:
        using Score::score;
        using Score::local_score;
        double score(const Model& m) const override {
            return static_cast<const Derived*>(this)->score(m);
        }
        double local_score(const Model& m, const int variable) const override {
            return static_cast<const Derived*>(this)->local_score(m, variable);
        }
        double local_score(const Model& m, const std::string& variable) const override {
            return static_cast<const Derived*>(this)->local_score(m, variable);
        }
        double local_score(const Model& m, 
                                   const int variable, 
                                   const typename std::vector<int>::const_iterator evidence_begin, 
                                   const typename std::vector<int>::const_iterator evidence_end) const override {
            return static_cast<const Derived*>(this)->local_score(m, variable, evidence_begin, evidence_end);
        }
        double local_score(const Model& m, 
                                   const std::string& variable, 
                                   const typename std::vector<std::string>::const_iterator evidence_begin, 
                                   const typename std::vector<std::string>::const_iterator evidence_end) const override {
            return static_cast<const Derived*>(this)->local_score(m, variable, evidence_begin, evidence_end);
        }
    };

    template<typename Derived, typename Model, typename... Models>
    class ScoreImpl<Derived, Model, Models...> : public ScoreImpl<Derived, Models...> {
    public:
        using ScoreImpl<Derived, Models...>::score;
        using ScoreImpl<Derived, Models...>::local_score;
        double score(const Model& m) const override {
            return static_cast<const Derived*>(this)->score(m);
        }
        double local_score(const Model& m, const int variable) const override {
            return static_cast<const Derived*>(this)->local_score(m, variable);
        }
        double local_score(const Model& m, const std::string& variable) const override {
            return static_cast<const Derived*>(this)->local_score(m, variable);
        }
        double local_score(const Model& m, 
                            const int variable, 
                            const typename std::vector<int>::const_iterator evidence_begin, 
                            const typename std::vector<int>::const_iterator evidence_end) const override {
            return static_cast<const Derived*>(this)->local_score(m, variable, evidence_begin, evidence_end);
        }
        double local_score(const Model& m, 
                            const std::string& variable, 
                            const typename std::vector<std::string>::const_iterator evidence_begin, 
                            const typename std::vector<std::string>::const_iterator evidence_end) const override {
            return static_cast<const Derived*>(this)->local_score(m, variable, evidence_begin, evidence_end);
        }
    };
}


#endif //PGM_DATASET_SCORES_HPP