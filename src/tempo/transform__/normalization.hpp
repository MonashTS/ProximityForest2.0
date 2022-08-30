#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>

namespace tempo::transform {

  namespace {

    // MeanNorm normalisation
    // avgv == avg(A) average value over A
    // minv == min(A) min value of A
    // maxv == max(A) max value of A
    // For constant series (maxv-minv==0), return a copy of A unchanged
    inline arma::Row<F> _meannorm(arma::Row<F> const& A, F avgv, F minv, F maxv) {
      F diffv = maxv - minv;
      if (diffv==0) { return A; }
      else { return (A - avgv)/diffv; }
    }

    // MinMax normalisation
    // minv == min(A)
    // maxv == max(A)
    // result within [range_min, range_max]
    // For constant series (maxv-minv==0), return middle of the range
    inline arma::Row<F> _minmax(arma::Row<F> const& A, F minv, F maxv, F range_min, F range_max) {
      F diffv = maxv - minv;
      F diffr = range_max - range_min;
      if (diffv==0) { return arma::Row<F>(A.n_elem, arma::fill::value(diffr/2)); }
      else { return range_min + (A - minv)*diffr/diffv; }
    }

    // ZScore normalisation
    // avgv == avg(A) average value over A
    // stdv == std(A) standard deviation value over A
    // For constant series (stdv==0), return a copy of A unchanged
    inline arma::Row<F> _zscore(arma::Row<F> const& A, F avgv, F stdv) {
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
  arma::Row<F> minmax(arma::Row<F> const& A, F range_min = 0, F range_max = 1);

  /// Normalisation MinMax for univariate TSeries
  /// By default, normalise in the 0-1 range.
  ///
  ///             (A - min(A)) * (range_max - range_min)
  /// range_min + --------------------------------------
  ///                    max(A) - min(A)
  ///
  /// If max(A) - min(A) = 0, returns a constant series, with same length as A, in the middle if the range
  TSeries minmax(TSeries const& A, F range_min = 0, F range_max = 1);

  /// Normalisation PercentileMinMax for arma::Row vector
  /// By default, normalise in the 0-1 range.
  /// minp = p percentile
  /// maxp = 100-p percentile
  ///
  ///             (A - minp) * (range_max - range_min)
  /// range_min + --------------------------------------
  ///                    maxp - minp
  ///
  /// If max(A) - min(A) = 0, returns a constant series, with same length as A, in the middle if the range
  /// 0<=p<50
  arma::Row<F> percentile_minmax(arma::Row<F> const& A, size_t p, F range_min, F range_max);

  TSeries percentile_minmax(TSeries const& A, size_t p, F range_min, F range_max);

  /// Normalisation MeanNorm for arma::Row vector
  ///
  ///    A - average(A)
  ///  ---------------------
  ///    max(A) - min(A)
  ///
  /// If max(A) - min(A) = 0, returns A
  arma::Row<F> meannorm(arma::Row<F> const& A);

  /// Normalisation MeanNorm for univariate TSeries
  ///
  ///    A - average(A)
  ///  ---------------------
  ///    max(A) - min(A)
  ///
  /// If max(A) - min(A) = 0, returns A
  TSeries meannorm(TSeries const& A);

  /// Normalisation UnitLength for arma::Row vector
  ///
  ///      A
  ///  ----------
  ///    ||A||        (norm of A)
  ///
  ///  If ||A|| = 0, returns A
  arma::Row<F> unitlength(arma::Row<F> const& A);

  /// Normalisation UnitLength for univariate TSeries
  ///
  ///      A
  ///  ----------
  ///    ||A||        (norm of A)
  ///
  ///  If ||A|| = 0, returns A
  TSeries unitlength(TSeries const& A);

  /// Normalisation Z-score for arma::Row vector
  ///
  ///    A - average(A)
  ///  ---------------------
  ///      stddev(A)
  ///
  /// If stddev(A) = 0, returns A
  arma::Row<F> zscore(arma::Row<F> const& A);

  /// Normalisation Z-score for arma::Row vector
  ///
  ///    A - average(A)
  ///  ---------------------
  ///      stddev(A)
  ///
  /// If stddev(A) = 0, returns A
  TSeries zscore(TSeries const& A);

}