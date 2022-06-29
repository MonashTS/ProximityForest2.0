#include "lockstep.hpp"

namespace tempo::distance {

  F lorentzian(arma::Row<F> const& A, arma::Row<F> const& B) {
    return arma::sum(arma::log(1 + arma::abs(A - B)));
  }

  F lorentzian(TSeries const& A, TSeries const& B) {
    arma::Row<F> a = A.rowvec();
    arma::Row<F> b = B.rowvec();
    return lorentzian(a, b);
  }

  F minkowski(arma::Row<F> const& A, arma::Row<F> const& B, F p) {
    return std::pow(arma::sum(arma::pow(arma::abs(A - B), p)), (F)1.0/p);
  }

  F minkowski(TSeries const& A, TSeries const& B, F p) {
    arma::Row<F> a = A.rowvec();
    arma::Row<F> b = B.rowvec();
    return minkowski(a, b, p);
  }

  F manhattan(arma::Row<F> const& A, arma::Row<F> const& B) {
    return arma::sum(arma::abs(A - B));
  }

  F manhattan(TSeries const& A, TSeries const& B) {
    arma::Row<F> a = A.rowvec();
    arma::Row<F> b = B.rowvec();
    return manhattan(a, b);
  }

} // End of namespace::distance
