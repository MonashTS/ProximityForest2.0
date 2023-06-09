#pragma once

#include <cstddef>

#include "core/univariate.derivative.hpp"
#include "core/univariate.normalization.hpp"
#include "core/univariate.noise.hpp"

namespace tempo::transform::univariate {

  // --- --- --- Derivative

  template<typename F>
  void derive(F const* data, size_t length, F* output){
    tempo::transform::core::univariate::derive<F, F const*, F*>(data, length, output);
  }

  template<typename F>
  void derive(F const* data, size_t length, F* output, size_t degree){
    if(degree==0){
      std::copy(data, data + length, output);
    } else {
      // D1
      derive(data, length, output);

      // D2 and after: need a temporary variable - derivative not computed "in place"
      if(degree>1){
        std::unique_ptr<F[]> tmp(new F[length]);
        F* a = output;
        F* b = tmp.get();

        for(size_t d=2; d<=degree; ++d){
          derive(a, length, b);
          std::swap(a, b);
        }

        // Copy in the output evey two turns
        if(degree%2==0){
          assert(b==output);
          std::copy(a, a+length, b);
        }
      }
    }
  }

  // --- --- --- Noise

  template<typename F, typename PRNG>
  void noise(F const* data, size_t length, F stddev, F delta, PRNG& prng, F* output){
    tempo::transform::core::univariate::noise<F, F const*, PRNG, F*>(data, length, stddev, delta, prng, output);
  }

  // --- --- --- Normalisation

  template<typename F>
  void minmax(F const* data, size_t length, F* output, F range_min, F range_max){
    tempo::transform::core::univariate::minmax<F>(data, length, output, range_min, range_max);
  }

  template<typename F>
  arma::Row<F> minmax(arma::Row<F> const& A, F range_min, F range_max) {
    return tempo::transform::core::univariate::minmax<F>(A, range_min, range_max);
  }

  template<typename F>
  void percentile_minmax(F const *data, size_t length, F *output, size_t p, F range_min, F range_max) {
    tempo::transform::core::univariate::percentile_minmax<F>(data, length, output, p, range_min, range_max);
  }

  template<typename F>
  arma::Row<F> percentile_minmax(arma::Row<F> const& A, size_t p, F range_min, F range_max) {
    return tempo::transform::core::univariate::percentile_minmax(A, p, range_min, range_max);
  }

  template<typename F>
  void meannorm(F const *data, size_t length, F *output){
    tempo::transform::core::univariate::meannorm(data, length, output);
  }

  template<typename F>
  arma::Row<F> meannorm(arma::Row<F> const& A) {
    return tempo::transform::core::univariate::meannorm(A);
  }

  template<typename F>
  void unitlength(F const *data, size_t length, F *output){
    tempo::transform::core::univariate::unitlength(data, length, output);
  }

  template<typename F>
  arma::Row<F> unitlength(arma::Row<F> const& A) {
    return tempo::transform::core::univariate::unitlength(A);
  }

  template<typename F>
  void zscore(F const *data, size_t length, F *output){
    tempo::transform::core::univariate::zscore(data, length, output);
  }

  template<typename F>
  arma::Row<F> zscore(arma::Row<F> const& A) {
    return tempo::transform::core::univariate::zscore(A);
  }

} // End of namespace tempo::transform::univariate
