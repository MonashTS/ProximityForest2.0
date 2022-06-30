#pragma once

#include <tempo/utils/utils.hpp>

namespace tempo::distance {

  namespace univariate {

    /// CFunBuilder Univariate Absolute difference exponent 1
    template<Subscriptable D>
    auto ad1(const D& lines, const D& cols) {
      return [&](size_t i, size_t j) {
        const F d = lines[i] - cols[j];
        return std::abs(d);
      };
    }

    /// CFunBuilder Univariate Absolute difference exponent 2
    template<Subscriptable D>
    auto ad2(const D& lines, const D& cols) {
      return [&](size_t i, size_t j) {
        const F d = lines[i] - cols[j];
        return d*d;
      };
    }

    /// CFunBuilder Univariate Absolute difference exponent e
    template<Subscriptable D>
    auto ade(const F e) {
      return [e](const D& lines, const D& cols) {
        return [&, e](size_t i, size_t j) {
          const F d = std::abs(lines[i] - cols[j]);
          return std::pow(d, e);
        };
      };
    }
  }

  namespace multivariate {

    /// CFunBuilder Multivariate Absolute difference exponent 2
    template<Subscriptable D>
    auto ad2N(const D& lines, const D& cols, size_t ndim) {
      return [&, ndim](size_t i, size_t j) {
        const size_t li_offset = i*ndim;
        const size_t co_offset = j*ndim;
        F acc{0};
        for (size_t k{0}; k<ndim; ++k) {
          const auto d = lines[li_offset + k] - cols[co_offset + k];
          acc += d*d;
        }
        return acc;
      };
    }
  }

  namespace WR {

    /// Returns both a cost and the window validity - for distance parameterized with a warping window.
    /// A cost of +Infinity means "early abandoned"
    /// Windows validity: given a distance (such as DTW, ERP and LCSS),
    /// represents the smallest window giving the same results for all other parameters being equals.
    struct WarpingResult {
      F cost;
      size_t window_validity;

      inline WarpingResult() : cost(tempo::utils::PINF), window_validity(tempo::utils::NO_WINDOW) {}

      inline WarpingResult(F c, size_t wv) : cost(c), window_validity(wv) {}

    };

  } // End of namespace WR

} // Enf of namespace tempo::distance
