
#include "tseries.univariate.hpp"

#include <tempo/dataset/tseries.hpp>
#include "core/univariate.normalization.hpp"
#include "core/univariate.derivative.hpp"

// Specialised implementation for TSeries
// When possible, use the computed stat per TSeries (like the min/max values) instead of recomputing them.

namespace tempo::transform::univariate {

  // --- --- --- Derivative

  TSeries derive(TSeries const& ts){
    const size_t l = ts.length();
    std::vector<F> d(l);
    double* data = d.data();
    tempo::transform::core::univariate::derive<F, F const*, F*>(ts.rawdata(), l, data);
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

} // End of namespace namespace tempo::transform::univariate
