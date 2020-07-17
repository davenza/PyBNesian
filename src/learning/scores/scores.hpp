#ifndef PGM_DATASET_SCORES_HPP
#define PGM_DATASET_SCORES_HPP

#include <graph/dag.hpp>
#include <models/GaussianNetwork.hpp>
#include <models/SemiparametricBN.hpp>

using graph::AdjListDag;
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

    template<typename... Types>
    class ScoreInterface {};


    template<typename Type>
    class ScoreInterface<Type> {
    public:
        inline virtual double score(const Type& m) const;
        inline virtual double local_score(const Type& m, const int) const;
        inline virtual double local_score(const Type& m, const std::string&) const;
        inline virtual double local_score(const Type& m, 
                                            const int, 
                                            const typename std::vector<int>::iterator, 
                                            const typename std::vector<int>::iterator) const;
        inline virtual double local_score(const Type& m, 
                                            const std::string&, 
                                            const typename std::vector<std::string>::iterator, 
                                            const typename std::vector<std::string>::iterator) const;
    };

    template<typename Type, typename... Types>
    class ScoreInterface<Type, Types...> : public ScoreInterface<Types...> {
    public:
        inline virtual double score(const Type& m) const;
        inline virtual double local_score(const Type& m, const int) const;
        inline virtual double local_score(const Type& m, const std::string&) const;
        inline virtual double local_score(const Type& m, 
                                            const int, 
                                            const typename std::vector<int>::iterator, 
                                            const typename std::vector<int>::iterator) const;
        inline virtual double local_score(const Type& m, 
                                            const std::string&, 
                                            const typename std::vector<std::string>::iterator, 
                                            const typename std::vector<std::string>::iterator) const;
    };

    class Score : public ScoreInterface<GaussianNetwork<>, 
                                        GaussianNetwork<AdjListDag>, 
                                        SemiparametricBN<>, 
                                        SemiparametricBN<AdjListDag>> {
    public:
        virtual bool is_decomposable() const = 0;
        virtual ScoreType type() const = 0;
    };


    template<typename Type>
    double ScoreInterface<Type>::score(const Type& m) const {
        throw std::invalid_argument("Score::score() not implemented for model " + m.type().ToString());
    }

    template<typename Type>
    double ScoreInterface<Type>::local_score(const Type& m, const int) const {
        throw std::invalid_argument("Score::local_score() not implemented for model " + m.type().ToString());
    }

    template<typename Type>
    double ScoreInterface<Type>::local_score(const Type& m, const std::string&) const {
        throw std::invalid_argument("Score::local_score() not implemented for  model " + m.type().ToString());
    }

    template<typename Type>
    double ScoreInterface<Type>::local_score(const Type& m, 
                                                const int, 
                                                const typename std::vector<int>::iterator, 
                                                const typename std::vector<int>::iterator) const {
        throw std::invalid_argument("Score::local_score() not implemented for model " + m.type().ToString());
    }

    template<typename Type>
    double ScoreInterface<Type>::local_score(const Type& m, 
                                                const std::string&, 
                                                const typename std::vector<std::string>::iterator, 
                                                const typename std::vector<std::string>::iterator) const {
        throw std::invalid_argument("Score::local_score() not implemented for model " + m.type().ToString());
    }

    template<typename Type, typename... Types>
    double ScoreInterface<Type, Types...>::score(const Type& m) const {
        throw std::invalid_argument("Score::score() not implemented for model " + m.type().ToString());
    }

    template<typename Type, typename... Types>
    double ScoreInterface<Type, Types...>::local_score(const Type& m, const int) const {
        throw std::invalid_argument("Score::local_score() not implemented for model " + m.type().ToString());
    }

    template<typename Type, typename... Types>
    double ScoreInterface<Type, Types...>::local_score(const Type& m, const std::string&) const {
        throw std::invalid_argument("Score::local_score() not implemented for model " + m.type().ToString());
    }

    template<typename Type, typename... Types>
    double ScoreInterface<Type, Types...>::local_score(const Type& m, 
                                                        const int, 
                                                        const typename std::vector<int>::iterator, 
                                                        const typename std::vector<int>::iterator) const {
        throw std::invalid_argument("Score::local_score() not implemented for model " + m.type().ToString());
    }

    template<typename Type, typename... Types>
    double ScoreInterface<Type, Types...>::local_score(const Type& m, 
                                                        const std::string&, 
                                                        const typename std::vector<std::string>::iterator, 
                                                        const typename std::vector<std::string>::iterator) const {
        throw std::invalid_argument("Score::local_score() not implemented for model " + m.type().ToString());
    }

    template<typename Derived, typename... Types>
    class ScoreImpl {};

    template<typename Derived, typename Type>
    class ScoreImpl<Derived, Type> : public Score {
    public:
        inline double score(const Type& m) const override {
            return static_cast<const Derived*>(this)->score(m);
        }

        inline double local_score(const Type& m, const int variable) const override {
            return static_cast<const Derived*>(this)->local_score(m, variable);
        }

        inline double local_score(const Type& m, const std::string& variable) const override {
            return static_cast<const Derived*>(this)->local_score(m, variable);
        }

        inline double local_score(const Type& m, 
                                   const int variable, 
                                   const typename std::vector<int>::iterator evidence_begin, 
                                   const typename std::vector<int>::iterator evidence_end) const override {
            return static_cast<const Derived*>(this)->local_score(m, variable, evidence_begin, evidence_end);
        }

        inline double local_score(const Type& m, 
                                   const std::string& variable, 
                                   const typename std::vector<std::string>::iterator evidence_begin, 
                                   const typename std::vector<std::string>::iterator evidence_end) const override {
            return static_cast<const Derived*>(this)->local_score(m, variable, evidence_begin, evidence_end);
        }
    };

    template<typename Derived, typename Type, typename... Types>
    class ScoreImpl<Derived, Type, Types...> : public ScoreImpl<Derived, Type>, public ScoreImpl<Derived, Types...> {
    public:
        inline double score(const Type& m) const override {
            return static_cast<const Derived*>(this)->score(m);
        }

        inline double local_score(const Type& m, const int variable) const override {
            return static_cast<const Derived*>(this)->local_score(m, variable);
        }

        inline double local_score(const Type& m, const std::string& variable) const override {
            return static_cast<const Derived*>(this)->local_score(m, variable);
        }

        inline double local_score(const Type& m, 
                                   const int variable, 
                                   const typename std::vector<int>::iterator evidence_begin, 
                                   const typename std::vector<int>::iterator evidence_end) const override {
            return static_cast<const Derived*>(this)->local_score(m, variable, evidence_begin, evidence_end);
        }

        inline double local_score(const Type& m, 
                                   const std::string& variable, 
                                   const typename std::vector<std::string>::iterator evidence_begin, 
                                   const typename std::vector<std::string>::iterator evidence_end) const override {
            return static_cast<const Derived*>(this)->local_score(m, variable, evidence_begin, evidence_end);
        }
    };



}


#endif //PGM_DATASET_SCORES_HPP