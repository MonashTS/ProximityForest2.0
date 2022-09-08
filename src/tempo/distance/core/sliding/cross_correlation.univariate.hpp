#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>

namespace tempo::distance::core::univariate {

  namespace cross_correlation {
    // Warning: when used in nearest neighbour search: higher score = more similarity (we want the furthest neighbour!)

    /// Cross Correlation sequence computation using FFT with a length adjusted to a power of 2 for efficiency.
    /// Based on https://github.com/johnpaparrizos/TSDistEval/blob/master/slidingmeasures/NCC.m
    template<typename F>
    arma::Row<F> cc_seq(arma::Row<F> const& A, arma::Row<F> const& B) {
      size_t len = std::max(A.n_elem, B.n_elem);
      size_t fftlenght = 1 << ::tempo::utils::nextpow2<size_t>(2*len - 1); // 1<<p  == 2^p for unsigned integral
      arma::Row<F> r = arma::real(arma::ifft(arma::fft(A, fftlenght)%arma::conj(arma::fft(B, fftlenght))));
      return arma::join_rows(r.cols(fftlenght - len + 1, fftlenght - 1), r.cols(0, len - 1));
    }

    /// Maximum Cross Correlation value
    template<typename F>
    F max_cc(arma::Row<F> const& A, arma::Row<F> const& B) {
      return arma::max(cc_seq(A, B));
    }

    /// Maximum Biased Normalized Cross Correlation value
    /// Divides the result of 'max_cc(A,B)' by max(len(A), len(B)) (max vector length)
    template<typename F>
    F max_ncc_b(arma::Row<F> const& A, arma::Row<F> const& B) {
      F n = std::max<F>(A.n_elem, B.n_elem);
      return arma::max(cc_seq(A, B))/n;
    }

    /// Maximum Coefficient Normalized Cross Correlation value
    /// Divides the result of 'max_cc(A,B)' by n = ||A||*||B|| (multiplication of the norms)
    /// Return a value in the [-1, 1] range, or NaN if a series is constant 0
    template<typename F>
    F max_ncc_c(arma::Row<F> const& A, arma::Row<F> const& B) {
      F n = arma::norm(A)*arma::norm(B);
      return arma::max(cc_seq(A, B))/n;
    }

  } // End of namespace cross_correlation

  /// Shape-based distance (sbd)
  /// k-Shape: Efficient and Accurate Clustering of Time Series, John Paparrizos & Luis Gravano, SIGMOD 2015
  /// returns  1 - max_ncc_c(A, B)
  /// Result is in [0, 2], "with 0 indicating perfect similarity".
  /// Warning: due to numerical approximation, sbd(A, A) returns a value close to 0 but not 0!
  /// If a series is constant 0, max_ncc_c returns NaN, and we return the max distance of 2.
  template<typename F>
  F sbd(arma::Row<F> const& A, arma::Row<F> const& B) {
    double cc = cross_correlation::max_ncc_c(A, B);
    if (std::isnan(cc)) { return 2.0; }
    return ((F)1) - cc;
  }

  template<typename F>
  F sbd(F const *A, size_t lA, F const *B, size_t lB) {
    const arma::Row<F> ra(const_cast<F *>(A), lA, false, true);
    const arma::Row<F> rb(const_cast<F *>(B), lB, false, true);
    return sbd<F>(ra, rb);
  }

} // End of namespace tempo::distance