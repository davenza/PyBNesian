#ifndef PGM_DATASET_SCORES_HPP
#define PGM_DATASET_SCORES_HPP


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
        constexpr bool operator!=(ScoreType a) const { return value != a.value; }

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

}


#endif //PGM_DATASET_SCORES_HPP