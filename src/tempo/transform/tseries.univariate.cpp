
#include "tseries.univariate.hpp"

#include <tempo/dataset/tseries.hpp>
#include "core/univariate.normalization.hpp"
#include "core/univariate.derivative.hpp"
#include "core/univariate.noise.hpp"
#include "core/univariate.frequency.hpp"
#include "univariate.hpp"

// Specialised implementation for TSeries
// When possible, use the computed stat per TSeries (like the min/max values) instead of recomputing them.

namespace tempo::transform::univariate {

  // --- --- --- Derivative

  TSeries derive(TSeries const& ts){
    const size_t l = ts.length();
    std::vector<F> d(l);
    double* data = d.data();
    tempo::transform::univariate::derive(ts.data(), l, data);
    return TSeries::mk_from_rowmajor(ts, std::move(d));
  }

  TSeries derive(TSeries const& ts, size_t degree){
    const size_t l = ts.length();
    std::vector<F> d(l);
    double* data = d.data();
    tempo::transform::univariate::derive(ts.data(), l, data, degree);
    return TSeries::mk_from_rowmajor(ts, std::move(d));
  }

  // --- --- --- Noise

  TSeries noise(TSeries const& ts, F delta, PRNG& prng){
    const size_t l = ts.length();
    const F stddev = ts.stddev()[0];
    std::vector<F> d(l);
    double* data = d.data();
    tempo::transform::univariate::noise(ts.data(), l, stddev, delta, prng, data);
    return TSeries::mk_from_rowmajor(ts, std::move(d));
  }

  // --- --- --- Normalisation

  TSeries minmax(TSeries const& A, F range_min, F range_max) {
    F minv = A.min()[0];
    F maxv = A.max()[0];
    arma::Row<F> v = tempo::transform::core::univariate::details::minmax(A.rowvec(), minv, maxv, range_min, range_max);
    return TSeries::mk_from(A, std::move(v));
  }

  TSeries percentile_minmax(TSeries const& A, size_t p, F range_min, F range_max) {
    arma::Row<F> v = tempo::transform::core::univariate::percentile_minmax(A.rowvec(), p, range_min, range_max);
    return TSeries::mk_from(A, std::move(v));
  }

  TSeries meannorm(TSeries const& A) {
    F minv = A.min()[0];
    F maxv = A.max()[0];
    F avgv = A.mean()[0];
    arma::Row<F> v = tempo::transform::core::univariate::details::meannorm(A.rowvec(), avgv, minv, maxv);
    return TSeries::mk_from(A, std::move(v));
  }

  TSeries unitlength(TSeries const& A) {
    arma::Row<F> v = tempo::transform::core::univariate::unitlength(A.rowvec());
    return TSeries::mk_from(A, std::move(v));
  }

  TSeries zscore(TSeries const& A) {
    F avgv = A.mean()[0];
    F stdv = A.stddev()[0];
    arma::Row<F> v = tempo::transform::core::univariate::details::zscore(A.rowvec(), avgv, stdv);
    return TSeries::mk_from(A, std::move(v));
  }

  // --- --- --- Frequency
  TSeries freqtransform(TSeries const& ts){
      const size_t l = ts.length();
      std::vector<F> d(l);
      double* data = d.data();
      tempo::transform::core::univariate::freqtransform<F, F const *, F *>(ts.data(), l, data);
      return TSeries::mk_from_rowmajor(ts, std::move(d));
  }

} // End of namespace namespace tempo::transform::univariate
