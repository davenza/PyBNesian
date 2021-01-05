#ifndef PYBNESIAN_UTIL_VIRTUAL_CLONE_HPP
#define PYBNESIAN_UTIL_VIRTUAL_CLONE_HPP

// Extracted from: https://www.fluentcpp.com/2017/09/12/how-to-return-a-smart-pointer-and-use-covariance/

namespace util {
///////////////////////////////////////////////////////////////////////////////
 
    template <typename T>
    class abstract_class { };
 
///////////////////////////////////////////////////////////////////////////////
 
    template <typename T>
    class virtual_inherit_from : virtual public T {
        using T::T;
    };
 
///////////////////////////////////////////////////////////////////////////////
 
    template <typename Derived, typename ... Bases>
    class clone_inherit : public Bases... {
    public:
        virtual ~clone_inherit() = default;

        std::unique_ptr<Derived> clone() const {
            return std::unique_ptr<Derived>(static_cast<Derived*>(this->clone_impl()));
        }
    protected:
   //         desirable, but impossible in C++17
   //         see: http://cplusplus.github.io/EWG/ewg-active.html#102
        using Bases::Bases...;
    private:
        virtual clone_inherit* clone_impl() const override {
            return new Derived(static_cast<const Derived & >(*this));
        }
    };
 
// ///////////////////////////////////////////////////////////////////////////////
 
    template <typename Derived, typename ... Bases>
    class clone_inherit<abstract_class<Derived>, Bases...> : public Bases... {
    public:
        virtual ~clone_inherit() = default;

        std::unique_ptr<Derived> clone() const {
            return std::unique_ptr<Derived>(static_cast<Derived*>(this->clone_impl()));
        }
    protected:
   //         desirable, but impossible in C++17
   //         see: http://cplusplus.github.io/EWG/ewg-active.html#102
        // using typename... Bases::Bases;
        using Bases::Bases...;
    private:
        virtual clone_inherit* clone_impl() const = 0;
    };
 
///////////////////////////////////////////////////////////////////////////////
 
    template <typename Derived>
    class clone_inherit<Derived> {
    public:
        virtual ~clone_inherit() = default;

        std::unique_ptr<Derived> clone() const {
            return std::unique_ptr<Derived>(static_cast<Derived*>(this->clone_impl()));
        }
    private:
        virtual clone_inherit* clone_impl() const override {
            return new Derived(static_cast<const Derived&>(*this));
        }
    };
 
///////////////////////////////////////////////////////////////////////////////
 
    template <typename Derived>
    class clone_inherit<abstract_class<Derived>> {
    public:
        virtual ~clone_inherit() = default;

        std::unique_ptr<Derived> clone() const {
            return std::unique_ptr<Derived>(static_cast<Derived*>(this->clone_impl()));
        }
    private:
        virtual clone_inherit* clone_impl() const = 0;
    };
 
///////////////////////////////////////////////////////////////////////////////
}

#endif //PYBNESIAN_UTIL_VIRTUAL_CLONE_HPP