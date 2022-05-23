#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>

namespace tempo::distance {

  /// Minkowski metric on arma row vector
  ///  - Equal to the Manhattan distance with p=1
  ///  - Equal to the Euclidean Distance distance with p=2
  template<Float F>
  F minkowski(arma::Row<F> const& A, arma::Row<F> const& B, F p) {
    return std::pow(arma::sum(arma::pow(arma::abs(A - B), p)), (F)1.0/p);
  }

  /// Minkowski metric on TSeries (univariate only)
  ///  - Equal to the Manhattan distance with p=1
  ///  - Equal to the Euclidean Distance distance with p=2
  inline F minkowski(TSeries const& A, TSeries const& B, F p) {
    arma::Row<F> a = A.rowvec();
    arma::Row<F> b = B.rowvec();
    return minkowski<F>(a, b, p);
  }

} // End of namespace::distance