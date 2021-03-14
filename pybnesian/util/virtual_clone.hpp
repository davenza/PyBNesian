#ifndef PYBNESIAN_UTIL_VIRTUAL_CLONE_HPP
#define PYBNESIAN_UTIL_VIRTUAL_CLONE_HPP

// Extracted from: https://www.fluentcpp.com/2017/09/12/how-to-return-a-smart-pointer-and-use-covariance/

namespace util {
///////////////////////////////////////////////////////////////////////////////

template <typename T>
class abstract_class {};

///////////////////////////////////////////////////////////////////////////////

template <typename T>
class virtual_inherit_from : virtual public T {
    using T::T;
};

///////////////////////////////////////////////////////////////////////////////

template <typename... Bases>
class add_constructors;

template <>
class add_constructors<> {};

template <typename Base>
class add_constructors<Base> : public Base {
public:
    using Base::Base;
};

template <typename Base, typename... Bases>
class add_constructors<Base, Bases...> : public Base, public add_constructors<Bases...> {
public:
    using Base::Base;
    using add_constructors<Bases...>::add_constructors;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Derived, typename... Bases>
class clone_inherit : public add_constructors<Bases...> {
public:
    virtual ~clone_inherit() = default;

    std::shared_ptr<Derived> clone() const {
        return std::shared_ptr<Derived>(static_cast<Derived*>(this->clone_impl()));
    }

protected:
    //         desirable, but impossible in C++17
    //         see: http://cplusplus.github.io/EWG/ewg-active.html#102
    //         This is not valid in gcc, but works in clang.
    // using typename... Bases::Bases...;

    using add_constructors<Bases...>::add_constructors;

private:
    virtual clone_inherit* clone_impl() const override { return new Derived(static_cast<const Derived&>(*this)); }
};

///////////////////////////////////////////////////////////////////////////////

template <typename Derived, typename... Bases>
class clone_inherit<abstract_class<Derived>, Bases...> : public add_constructors<Bases...> {
public:
    virtual ~clone_inherit() = default;

    std::shared_ptr<Derived> clone() const {
        return std::shared_ptr<Derived>(static_cast<Derived*>(this->clone_impl()));
    }

protected:
    //         desirable, but impossible in C++17
    //         see: http://cplusplus.github.io/EWG/ewg-active.html#102
    // using typename... Bases::Bases...;

    using add_constructors<Bases...>::add_constructors;

private:
    virtual clone_inherit* clone_impl() const = 0;
};

///////////////////////////////////////////////////////////////////////////////

template <bool cloneable, typename Derived, typename... Bases>
class clone_inherit_condition;

template <typename Derived, typename... Bases>
class clone_inherit_condition<true, Derived, Bases...> : public add_constructors<Bases...> {
public:
    virtual ~clone_inherit_condition() = default;

    std::shared_ptr<Derived> clone() const {
        return std::shared_ptr<Derived>(static_cast<Derived*>(this->clone_impl()));
    }

protected:
    //         desirable, but impossible in C++17
    //         see: http://cplusplus.github.io/EWG/ewg-active.html#102
    //         This is not valid in gcc, but works in clang.
    // using typename... Bases::Bases...;

    using add_constructors<Bases...>::add_constructors;

private:
    virtual clone_inherit_condition* clone_impl() const override {
        return new Derived(static_cast<const Derived&>(*this));
    }
};

/////////////////////////////////////////////////////////////////////////////////

template <typename Derived, typename... Bases>
class clone_inherit_condition<true, abstract_class<Derived>, Bases...> : public add_constructors<Bases...> {
public:
    virtual ~clone_inherit_condition() = default;

    std::shared_ptr<Derived> clone() const {
        return std::shared_ptr<Derived>(static_cast<Derived*>(this->clone_impl()));
    }

protected:
    //         desirable, but impossible in C++17
    //         see: http://cplusplus.github.io/EWG/ewg-active.html#102
    //         This is not valid in gcc, but works in clang.
    // using typename... Bases::Bases...;

    using add_constructors<Bases...>::add_constructors;

private:
    virtual clone_inherit_condition* clone_impl() const = 0;
};

/////////////////////////////////////////////////////////////////////////////////

template <typename Derived, typename... Bases>
class clone_inherit_condition<false, Derived, Bases...> : public add_constructors<Bases...> {
protected:
    //         desirable, but impossible in C++17
    //         see: http://cplusplus.github.io/EWG/ewg-active.html#102
    //         This is not valid in gcc, but works in clang.
    // using typename... Bases::Bases...;

    using add_constructors<Bases...>::add_constructors;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Derived>
class clone_inherit<Derived> {
public:
    virtual ~clone_inherit() = default;

    std::shared_ptr<Derived> clone() const {
        return std::shared_ptr<Derived>(static_cast<Derived*>(this->clone_impl()));
    }

private:
    virtual clone_inherit* clone_impl() const override { return new Derived(static_cast<const Derived&>(*this)); }
};

///////////////////////////////////////////////////////////////////////////////

template <typename Derived>
class clone_inherit<abstract_class<Derived>> {
public:
    virtual ~clone_inherit() = default;

    std::shared_ptr<Derived> clone() const {
        return std::shared_ptr<Derived>(static_cast<Derived*>(this->clone_impl()));
    }

private:
    virtual clone_inherit* clone_impl() const = 0;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Derived>
class clone_inherit_condition<true, Derived> {
public:
    virtual ~clone_inherit_condition() = default;

    std::shared_ptr<Derived> clone() const {
        return std::shared_ptr<Derived>(static_cast<Derived*>(this->clone_impl()));
    }

private:
    virtual clone_inherit_condition* clone_impl() const override {
        return new Derived(static_cast<const Derived&>(*this));
    }
};

///////////////////////////////////////////////////////////////////////////////

template <typename Derived>
class clone_inherit_condition<true, abstract_class<Derived>> {
public:
    virtual ~clone_inherit_condition() = default;

    std::shared_ptr<Derived> clone() const {
        return std::shared_ptr<Derived>(static_cast<Derived*>(this->clone_impl()));
    }

private:
    virtual clone_inherit_condition* clone_impl() const = 0;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Derived>
class clone_inherit_condition<false, Derived> {};

}  // namespace util

#endif  // PYBNESIAN_UTIL_VIRTUAL_CLONE_HPP