#pragma once

#include <tempo/dataset/tseries.hpp>

namespace tempo::transform::univariate {

  // --- --- --- Derivative

  /// Derivative, as defined in DDTW, for univariate TSeries
  TSeries derive(TSeries const& ts);

  /// n-th derivative
  TSeries derive(TSeries const& ts, size_t degree);

  // --- --- --- Noise

  /// Add noise n to each timestamp where n = delta*U(0, stddev of the series)
  TSeries noise(TSeries const& ts, F delta, PRNG& prng);

  // --- --- --- Normalisation

  /// Normalisation MinMax for univariate TSeries
  /// Usual default values are range_min = 0 and range_min = 1
  ///
  ///             (A - min(A)) * (range_max - range_min)
  /// range_min + --------------------------------------
  ///                    max(A) - min(A)
  ///
  /// If max(A) - min(A) = 0, returns a constant series, with same length as A, in the middle if the range
  TSeries minmax(TSeries const& A, F range_min, F range_max);

  /// Normalisation PercentileMinMax for Univaraite TSeries
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
  /// 0<=p<50
  TSeries percentile_minmax(TSeries const& A, size_t p, F range_min, F range_max);

  /// Normalisation MeanNorm for univariate TSeries
  ///
  ///    A - average(A)
  ///  ---------------------
  ///    max(A) - min(A)
  ///
  /// If max(A) - min(A) = 0, returns A
  TSeries meannorm(TSeries const& A);

  /// Normalisation UnitLength for univariate TSeries
  ///
  ///      A
  ///  ----------
  ///    ||A||        (norm of A)
  ///
  ///  If ||A|| = 0, returns A
  TSeries unitlength(TSeries const& A);

  /// Normalisation Z-score for univariate TSeries
  ///
  ///    A - average(A)
  ///  ---------------------
  ///      stddev(A)
  ///
  /// If stddev(A) = 0, returns A
  TSeries zscore(TSeries const& A);

} // End of namespace namespace tempo::transform::univariate
