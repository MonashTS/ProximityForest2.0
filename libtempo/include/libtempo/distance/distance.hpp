#pragma once

#include <libtempo/utils/utils.hpp>
#include <libtempo/utils/point.hpp>

#include <functional>
#include <variant>
#include <tuple>

namespace libtempo::distance {

  namespace internal {

    /// Squared Euclidean Distance dim 1
    template<typename FloatType, typename It>
    [[nodiscard]] inline FloatType sqed1(const It& lines, size_t li, const It& cols, size_t co) {
      FloatType d = lines[li]-cols[co];
      return d*d;
    }

    /// Squared Euclidean Distance dim N
    template<typename FloatType, typename It>
    [[nodiscard]] inline FloatType sqedN(const It& lines, size_t li, const It& cols, size_t co, size_t ndim) {
      const size_t li_offset = li*ndim;
      const size_t co_offset = co*ndim;
      FloatType acc{0};
      for (size_t k{0}; k<ndim; ++k) {
        FloatType d = lines[li_offset+k]-cols[co_offset+k];
        acc += d*d;
      }
      return acc;
    }

    /// Ensure that the series are correctly ordered, following the EAP order (longest along the lines)
    template<typename It>
    [[nodiscard]] inline auto order(size_t nbli, const It& s1, size_t nbco, const It& s2) {
      if (nbli>=nbco) { return std::tuple(s1, s2); } else { return std::tuple(s2, s1); }
    }


  } // End of namespace internal


  /// SQED1 raw builder - ensure that the series are correctly ordered according to
  template<typename FloatType, typename It>
  [[nodiscard]] inline auto sqed(size_t nbli, const It& s1, size_t nbco, const It& s2) {
    const auto& [lines, cols] = internal::order(nbli, s1, nbco, s2);
    return [lines, cols](size_t i, size_t j) { return internal::sqed1<FloatType, It>(lines, i, cols, j); };
  }

  /// SQED1 builder for vector like types
  template<typename FloatType>
  [[nodiscard]] inline auto sqed(const auto& s1, const auto& s2) {
    return sqed<FloatType>(s1.size(), s1, s2.size(), s2);
  }

  /// SQEDN raw builder
  template<typename FloatType, typename It>
  [[nodiscard]] inline auto sqed(size_t nbli, const It& s1, size_t nbco, const It& s2, size_t ndim) {
    const auto& [lines, cols] = internal::order(nbli, s1, nbco, s2);
    return [ndim, lines, cols](size_t i, size_t j) { return internal::sqedN<FloatType, It>(lines, i, cols, j, ndim); };
  }

  /// SQEDN builder for vector like types
  template<typename FloatType>
  [[nodiscard]] inline auto sqed(const auto& s1, const auto& s2, size_t ndim) {
    return sqed<FloatType>(s1.size(), s1, s2.size(), s2, ndim);
  }



  /// Type alias for a tuple representing (nblines, nbcols)
  using lico_t = std::tuple<size_t, size_t>;

  /// Helper function checking and ordering the length of the series
  template<typename FloatType>
  [[nodiscard]] inline std::variant<FloatType, lico_t> check_order_series(size_t length1, size_t length2) {
    // Check sizes. If both series are empty, return 0, else if one is empty and not the other, maximal error.
    if (length1==0 && length2==0) { return {FloatType(0.0)}; }
    else if ((length1==0)!=(length2==0)) { return utils::PINF<FloatType>; }
    // Use the smallest size as the columns (which will be the allocation size)
    return (length1>length2) ? std::tuple(length1, length2) : std::tuple(length2, length1);
  }

} // Enf of namespace libtempo::distance
