#include <boost/math/constants/constants.hpp>

namespace util {


    template<typename T>
    inline auto constexpr pi = boost::math::constants::pi<T>();
}