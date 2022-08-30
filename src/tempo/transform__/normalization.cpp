#include "normalization.hpp"


namespace tempo::transform {

  /// Normalisation MinMax for arma::Row vector
  /// By default, normalise in the 0-1 range.
  ///
  ///             (A - min(A)) * (range_max - range_min)
  /// range_min + --------------------------------------
  ///                    max(A) - min(A)
  ///
  /// If max(A) - min(A) = 0, returns a constant series, with same length as A, in the middle if the range
  arma::Row<F> minmax(arma::Row<F> const& A, F range_min, F range_max) {
    F minv = arma::min(A);
    F maxv = arma::max(A);
    return _minmax(A, minv, maxv, range_min, range_max);
  }

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
  arma::Row<F> percentile_minmax(arma::Row<F> const& A, size_t p, F range_min, F range_max) {
    assert(p<50);
    arma::Row<F> B = arma::sort(A);
    size_t s = A.size();
    size_t minp = (p*s)/100;
    size_t maxp = ((100-p)*s)/100;
    F minv = B[minp];
    F maxv = B[maxp];
    return _minmax(A, minv, maxv, range_min, range_max);
  }

  TSeries percentile_minmax(TSeries const& A, size_t p, F range_min, F range_max) {
    arma::Row<F> v = percentile_minmax(A.rowvec(), p, range_min, range_max);
    return TSeries::mk_from(A, std::move(v));
  }

  /// Normalisation MinMax for univariate TSeries
  /// By default, normalise in the 0-1 range.
  ///
  ///             (A - min(A)) * (range_max - range_min)
  /// range_min + --------------------------------------
  ///                    max(A) - min(A)
  ///
  /// If max(A) - min(A) = 0, returns a constant series, with same length as A, in the middle if the range
  TSeries minmax(TSeries const& A, F range_min, F range_max) {
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
  arma::Row<F> unitlength(arma::Row<F> const& A) {
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
  TSeries unitlength(TSeries const& A) {
    arma::Row<F> v = unitlength(A.rowvec());
    return TSeries::mk_from(A, std::move(v));
  }

  /// Normalisation Z-score for arma::Row vector
  ///
  ///    A - average(A)
  ///  ---------------------
  ///      stddev(A)
  ///
  /// If stddev(A) = 0, returns A
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

} // Enf of namespace tempo::transform
