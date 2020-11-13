#ifndef PGM_DATASET_UTIL_PROGRESS_HPP
#define PGM_DATASET_UTIL_PROGRESS_HPP

#include <indicators/progress_spinner.hpp>

namespace util {

    class BaseIndeterminateSpinner {
    public:
        virtual ~BaseIndeterminateSpinner() {}
        virtual void update_status(const std::string& s) = 0;
        virtual void update_status(std::string&& s) = 0;
        virtual void update_status() = 0;
        virtual void mark_as_completed(const std::string& s) = 0;
        virtual void mark_as_completed(std::string&& s) = 0;
        virtual void mark_as_completed() = 0;
    };

    class VoidProgress : public BaseIndeterminateSpinner {
    public:
        VoidProgress() = default;

        void update_status(const std::string&) override {}
        void update_status(std::string&&) override {}
        void update_status() override {}
        void mark_as_completed(const std::string&) override {}
        void mark_as_completed(std::string&&) override {}
        void mark_as_completed() override {}
    };

    class IndeterminateSpinner : public BaseIndeterminateSpinner{
    public:

        IndeterminateSpinner() : m_spinner(indicators::option::PostfixText{"Checking dataset..."},
                                           indicators::option::MaxPostfixTextLen{0},
                                           indicators::option::ShowPercentage{false},
                                           indicators::option::ShowElapsedTime{true},
                                           indicators::option::SpinnerStates{
                                                std::vector<std::string>{"⠈", "⠐", "⠠", "⢀", "⡀", "⠄", "⠂", "⠁"}},
                                           indicators::option::FontStyles{
                                               std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}) {}

        template<typename...Args>
        IndeterminateSpinner(Args&&... args) : m_spinner(args...) {}

        void update_status(const std::string& s) override {
            m_spinner.set_option(indicators::option::PostfixText{s});
            update_status();
        }
        void update_status(std::string&& s) override {
            update_status(s);
        }

        void update_status() override {
            m_spinner.set_progress(1);
        }

        void mark_as_completed(const std::string& s) override {
            m_spinner.set_option(indicators::option::PrefixText{"✔"});
            m_spinner.set_option(indicators::option::ShowSpinner{false});
            m_spinner.set_option(indicators::option::PostfixText{s});
            m_spinner.set_option(indicators::option::ForegroundColor{indicators::Color::green});
            mark_as_completed();
        }
        
        void mark_as_completed(std::string&& s) override {
            mark_as_completed(s);
        }

        void mark_as_completed() override {
            m_spinner.mark_as_completed();
        }
    private:
        indicators::ProgressSpinner m_spinner;
    };


    template<typename... Args>
    std::unique_ptr<BaseIndeterminateSpinner> indeterminate_spinner(int verbose_level, Args&&... additional_args) {
        switch (verbose_level) {
            case 0:
                return std::make_unique<VoidProgress>();
            case 1: {
                if constexpr (sizeof...(Args) == 0)
                    return std::make_unique<IndeterminateSpinner>();
                else
                    return std::make_unique<IndeterminateSpinner>(additional_args...);
            }
            default:
                throw std::invalid_argument("Wrong verbose level. Allowed values are 0 and 1.");

        }
    }
}

#endif //PGM_DATASET_UTIL_PROGRESS_HPP