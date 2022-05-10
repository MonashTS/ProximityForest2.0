#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/tseries/tseries.hpp>

namespace tempo::distance {

  /// Minkowski metric on arma::Row
  ///  - Equal to the Manhattan distance with p=1
  ///  - Equal to the Euclidean Distance distance with p=2
  template<Float F>
  F minkowski(arma::Row<F> const& A, arma::Row<F> const& B, F p, [[maybe_unused]]F ub = utils::PINF<F>) {
    if(p==1.0){ // Manhattan distance
      return arma::sum(arma::abs(A - B));
    } else {
      return std::pow(arma::sum(arma::pow(arma::abs(A - B), p)), (F)1.0/p);
    }
  }

  /// Minkowski metric on TSeries (univariate only)
  ///  - Equal to the Manhattan distance with p=1
  ///  - Equal to the Euclidean Distance distance with p=2
  inline F minkowski(TSeries const& A, TSeries const& B, F p, F ub = utils::PINF<F>) {
    arma::Row<F> a = A.rowvec();
    arma::Row<F> b = B.rowvec();
    return minkowski<F>(a, b, p, ub);
  }

} // End of namespace::distance