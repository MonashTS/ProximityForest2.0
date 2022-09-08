#pragma once

#include "tempo/utils/utils.hpp"
#include "tempo/dataset/tseries.hpp"

namespace tempo::transform::core::univariate {

  namespace details {

    // MeanNorm normalisation
    // avgv == avg(A) average value over A
    // minv == min(A) min value of A
    // maxv == max(A) max value of A
    // For constant series (maxv-minv==0), return a copy of A unchanged
    template<typename F>
    arma::Row<F> meannorm(arma::Row<F> const& A, F avgv, F minv, F maxv) {
      F diffv = maxv - minv;
      if (diffv==0) { return A; }
      else { return (A - avgv)/diffv; }
    }

    // MinMax normalisation
    // minv == min(A)
    // maxv == max(A)
    // result within [range_min, range_max]
    // For constant series (maxv-minv==0), return middle of the range
    template<typename F>
    arma::Row<F> minmax(arma::Row<F> const& A, F minv, F maxv, F range_min, F range_max) {
      F diffv = maxv - minv;
      F diffr = range_max - range_min;
      if (diffv==0) { return arma::Row<F>(A.n_elem, arma::fill::value(diffr/2)); }
      else { return range_min + (A - minv)*diffr/diffv; }
    }

    // ZScore normalisation
    // avgv == avg(A) average value over A
    // stdv == std(A) standard deviation value over A
    // For constant series (stdv==0), return a copy of A unchanged
    template<typename F>
    arma::Row<F> zscore(arma::Row<F> const& A, F avgv, F stdv) {
      if (stdv==0) { return A; }
      else { return (A - avgv)/stdv; }
    }
  }

  /// Normalisation MinMax
  /// Usual default values are range_min = 0 and range_min = 1
  ///
  ///             (A - min(A)) * (range_max - range_min)
  /// range_min + --------------------------------------
  ///                    max(A) - min(A)
  ///
  /// If max(A) - min(A) = 0, returns a constant series, with same length as A, in the middle if the range

  template<typename F>
  arma::Row<F> minmax(arma::Row<F> const& A, F range_min, F range_max) {
    F minv = arma::min(A);
    F maxv = arma::max(A);
    return details::minmax(A, minv, maxv, range_min, range_max);
  }

  template<typename F>
  void minmax(F const *data, size_t length, F *output, F range_min = 0, F range_max = 1) {
    // Wrap external memory: no copy (direct use, 1st flag at false) and no reallocation (strict, 2nd flag at true)
    // Need const cast as armadillo only support building from non-const pointer
    const arma::Row<F> rvec_in(const_cast<F *>(data), length, false, true);
    arma::Row<F> rvec_out(output, length, false, true);
    // ---
    rvec_out = minmax(rvec_in, range_min, range_max);
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /// Normalisation PercentileMinMax
  /// For p  0 <= p < 50
  /// minp = p percentile
  /// maxp = 100-p percentile
  /// A usual range is range_min=0 and range_max = 1
  ///
  ///             (A - minp) * (range_max - range_min)
  /// range_min + --------------------------------------
  ///                    maxp - minp
  ///
  /// If max(A) - min(A) = 0, returns a constant series, with same length as A, in the middle if the range
  template<typename F>
  arma::Row<F> percentile_minmax(arma::Row<F> const& A, size_t p, F range_min, F range_max) {
    assert(p<50);
    arma::Row<F> B = arma::sort(A);
    size_t s = A.size();
    size_t minp = (p*s)/100;
    size_t maxp = ((100 - p)*s)/100;
    F minv = B[minp];
    F maxv = B[maxp];
    return details::minmax(A, minv, maxv, range_min, range_max);
  }

  template<typename F>
  void percentile_minmax(F const *data, size_t length, F *output, size_t p, F range_min, F range_max) {
    // Wrap external memory: no copy (direct use, 1st flag at false) and no reallocation (strict, 2nd flag at true)
    // Need const cast as armadillo only support building from non-const pointer
    const arma::Row<F> rvec_in(const_cast<F *>(data), length, false, true);
    arma::Row<F> rvec_out(output, length, false, true);
    // ---
    rvec_out = percentile_minmax<F>(rvec_in, p, range_min, range_max);
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /// Normalisation MeanNorm
  ///
  ///    A - average(A)
  ///  ---------------------
  ///    max(A) - min(A)
  ///
  /// If max(A) - min(A) = 0, returns A
  template<typename F>
  arma::Row<F> meannorm(arma::Row<F> const& A) {
    F minv = arma::min(A);
    F maxv = arma::max(A);
    F avgv = arma::mean(A);
    return details::meannorm(A, avgv, minv, maxv);
  }

  template<typename F>
  void meannorm(F const *data, size_t length, F *output){
    // Wrap external memory: no copy (direct use, 1st flag at false) and no reallocation (strict, 2nd flag at true)
    // Need const cast as armadillo only support building from non-const pointer
    const arma::Row<F> rvec_in(const_cast<F *>(data), length, false, true);
    arma::Row<F> rvec_out(output, length, false, true);
    // ---
    rvec_out = meannorm<F>(rvec_in);
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---


  /// Normalisation UnitLength
  ///
  ///      A
  ///  ----------
  ///    ||A||        (norm of A)
  ///
  ///  If ||A|| = 0, returns A
  template<typename F>
  arma::Row<F> unitlength(arma::Row<F> const& A) {
    F norm = arma::norm(A);
    if (norm==0) { return A; } else { return A/norm; }
  }

  template<typename F>
  void unitlength(F const *data, size_t length, F *output){
    // Wrap external memory: no copy (direct use, 1st flag at false) and no reallocation (strict, 2nd flag at true)
    // Need const cast as armadillo only support building from non-const pointer
    const arma::Row<F> rvec_in(const_cast<F *>(data), length, false, true);
    arma::Row<F> rvec_out(output, length, false, true);
    // ---
    rvec_out = unitlength<F>(rvec_in);
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---


  /// Normalisation Z-score
  ///
  ///    A - average(A)
  ///  ---------------------
  ///      stddev(A)
  ///
  /// If stddev(A) = 0, returns A
  template<typename F>
  arma::Row<F> zscore(arma::Row<F> const& A) {
    F avgv = arma::mean(A);
    F stdv = arma::stddev(A);
    return details::zscore(A, avgv, stdv);
  }

  template<typename F>
  void zscore(F const *data, size_t length, F *output){
    // Wrap external memory: no copy (direct use, 1st flag at false) and no reallocation (strict, 2nd flag at true)
    // Need const cast as armadillo only support building from non-const pointer
    const arma::Row<F> rvec_in(const_cast<F *>(data), length, false, true);
    arma::Row<F> rvec_out(output, length, false, true);
    // ---
    rvec_out = zscore<F>(rvec_in);
  }

}