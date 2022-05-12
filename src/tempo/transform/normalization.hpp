#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/tseries/tseries.hpp>

namespace tempo::transform {

  namespace {

    // MinMax normalisation
    // minv == min(A)
    // maxv == max(A)
    // result within [range_min, range_max]
    // For constant series (maxv-minv==0), return middle of the range
    template<typename F>
    arma::Row<F> _minmax(arma::Row<F> const& A, F minv, F maxv, F range_min, F range_max) {
      F diffv = maxv - minv;
      F diffr = range_max - range_min;
      if (diffv==0) { return arma::Row<F>(A.n_elem, arma::fill::value(diffr/2)); }
      else { return range_min + (A - minv)*diffr/diffv; }
    }

    // MeanNorm normalisation
    // avgv == avg(A) average value over A
    // minv == min(A) min value of A
    // maxv == max(A) max value of A
    // For constant series (maxv-minv==0), return a copy of A unchanged
    template<typename F>
    arma::Row<F> _meannorm(arma::Row<F> const& A, F avgv, F minv, F maxv) {
      F diffv = maxv - minv;
      if (diffv==0) { return A; }
      else { return (A - avgv)/diffv; }
    }

    // ZScore normalisation
    // avgv == avg(A) average value over A
    // stdv == std(A) standard deviation value over A
    // For constant series (stdv==0), return a copy of A unchanged
    template<typename F>
    arma::Row<F> _zscore(arma::Row<F> const& A, F avgv, F stdv) {
      if (stdv==0) { return A; }
      else { return (A - avgv)/stdv; }
    }

  }

  /// Normalisation MinMax for arma::Row vector
  /// By default, normalise in the 0-1 range.
  ///
  ///             (A - min(A)) * (range_max - range_min)
  /// range_min + --------------------------------------
  ///                    max(A) - min(A)
  ///
  /// If max(A) - min(A) = 0, returns a constant series, with same length as A, in the middle if the range
  template<typename F>
  arma::Row<F> minmax(arma::Row<F> const& A, F range_min = 0, F range_max = 1) {
    F minv = arma::min(A);
    F maxv = arma::max(A);
    return _minmax(A, minv, maxv, range_min, range_max);
  }

  /// Normalisation MinMax for univariate TSeries
  /// By default, normalise in the 0-1 range.
  ///
  ///             (A - min(A)) * (range_max - range_min)
  /// range_min + --------------------------------------
  ///                    max(A) - min(A)
  ///
  /// If max(A) - min(A) = 0, returns a constant series, with same length as A, in the middle if the range
  TSeries minmax(TSeries const& A, F range_min = 0, F range_max = 1) {
    F minv = A.min()[0];
    F maxv = A.max()[0];
    arma::Row<F> v = _minmax(A.rowvec(), minv, maxv, range_min, range_max);
    return TSeries::mk_from(A, std::move(v));
  }

  /// Normalisation MeanNorm for arma::Row vector
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
    return _meannorm(A, avgv, minv, maxv);
  }

  /// Normalisation MeanNorm for univariate TSeries
  ///
  ///    A - average(A)
  ///  ---------------------
  ///    max(A) - min(A)
  ///
  /// If max(A) - min(A) = 0, returns A
  TSeries meannorm(TSeries const& A) {
    F minv = A.min()[0];
    F maxv = A.max()[0];
    F avgv = A.mean()[0];
    arma::Row<F> v = _meannorm(A.rowvec(), avgv, minv, maxv);
    return TSeries::mk_from(A, std::move(v));
  }

  /// Normalisation UnitLength for arma::Row vector
  ///
  ///      A
  ///  ----------
  ///    ||A||        (norm of A)
  ///
  ///  If ||A|| = 0, returns A
  template<typename F>
  arma::Row<F> unitlenght(arma::Row<F> const& A) {
    F norm = arma::norm(A);
    if (norm==0) { return A; } else { return A/norm; }
  }

  /// Normalisation UnitLength for univariate TSeries
  ///
  ///      A
  ///  ----------
  ///    ||A||        (norm of A)
  ///
  ///  If ||A|| = 0, returns A
  TSeries unitlenght(TSeries const& A) {
    arma::Row<F> v = unitlenght(A.rowvec());
    return TSeries::mk_from(A, std::move(v));
  }

  /// Normalisation Z-score for arma::Row vector
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
    return _zscore(A, avgv, stdv);
  }

  /// Normalisation Z-score for arma::Row vector
  ///
  ///    A - average(A)
  ///  ---------------------
  ///      stddev(A)
  ///
  /// If stddev(A) = 0, returns A
  TSeries zscore(TSeries const& A) {
    F avgv = A.mean()[0];
    F stdv = A.stddev()[0];
    arma::Row<F> v = _zscore(A.rowvec(), avgv, stdv);
    return TSeries::mk_from(A, std::move(v));
  }

}