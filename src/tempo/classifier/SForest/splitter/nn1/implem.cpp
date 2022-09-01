#include "sp_da.hpp"
#include <tempo/distance__/lockstep/direct.hpp>

#include "sp_dtw.hpp"
#include <tempo/distance__/elastic/dtw.hpp>

#include "sp_adtw.hpp"
#include <tempo/distance__/elastic/adtw.hpp>

#include "sp_wdtw.hpp"
#include <tempo/distance__/elastic/wdtw.hpp>

#include "sp_erp.hpp"
#include <tempo/distance__/elastic/erp.hpp>

#include "sp_lcss.hpp"
#include <tempo/distance__/elastic/lcss.hpp>

#include "sp_msm.hpp"
#include <tempo/distance__/elastic/msm.hpp>

#include "sp_twe.hpp"
#include <tempo/distance__/elastic/twe.hpp>

#include "sp_lorentzian.hpp"
#include <tempo/distance__/lockstep/lockstep.hpp>

#include "sp_sbd.hpp"
#include <tempo/distance__/sliding/cross_correlation.hpp>

namespace tempo::classifier::SForest::splitter::nn1 {

  namespace {
    constexpr auto d1 = distance::univariate::ad1<TSeries>;
    constexpr auto d2 = distance::univariate::ad2<TSeries>;
  }

  F DA::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    if (exponent==1.0) {
      return distance::directa(t1, t2, d1, bsf);
    } else if (exponent==2.0) {
      return distance::directa(t1, t2, d2, bsf);
    } else {
      return distance::directa(t1, t2, distance::univariate::ade<TSeries>(exponent), bsf);
    }
  }

  F DTW::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    if (exponent==1.0) {
      return distance::dtw(t1, t2, d1, w, bsf);
    } else if (exponent==2.0) {
      return distance::dtw(t1, t2, d2, w, bsf);
    } else {
      return distance::dtw(t1, t2, distance::univariate::ade<TSeries>(exponent), w, bsf);
    }
  }

  F ADTW::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    if (exponent==1.0) {
      return distance::adtw(t1, t2, d1, penalty, bsf);
    } else if (exponent==2.0) {
      return distance::adtw(t1, t2, d2, penalty, bsf);
    } else {
      return distance::adtw(t1, t2, distance::univariate::ade<TSeries>(exponent), penalty, bsf);
    }
  }

  F WDTW::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    if (exponent==1.0) {
      return distance::wdtw(t1, t2, d1, weights, bsf);
    } else if (exponent==2.0) {
      return distance::wdtw(t1, t2, d2, weights, bsf);
    } else {
      return distance::wdtw(t1, t2, distance::univariate::ade<TSeries>(exponent), weights, bsf);
    }
  }

  F ERP::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    if (exponent==1.0) {
      return distance::erp(t1, t2,
                           tempo::distance::univariate::ad1gv<TSeries>,
                           d1,
                           w, gv, bsf);
    } else if (exponent==2.0) {
      return distance::erp(t1, t2,
                           tempo::distance::univariate::ad2gv<TSeries>,
                           d2,
                           w, gv, bsf);
    } else {
      return distance::erp(t1, t2,
                           tempo::distance::univariate::adegv<TSeries>(exponent),
                           tempo::distance::univariate::ade<TSeries>(exponent),
                           w, gv, bsf);
    }
  }

  F LCSS::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    if (exponent==1.0) {
      return distance::lcss(t1, t2, d1, w, epsilon, bsf);
    } else if (exponent==2.0) {
      return distance::lcss(t1, t2, d2, w, epsilon, bsf);
    } else {
      return distance::lcss(t1, t2, distance::univariate::ade<TSeries>(exponent), w, epsilon, bsf);
    }
  }

  F MSM::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    return distance::univariate::msm(t1, t2, cost, bsf);
  }

  F TWE::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    return distance::univariate::twe(t1, t2, nu, lambda, bsf);
  }
//
//  F Lorentzian::eval(const TSeries& t1, const TSeries& t2, F /* bsf */) {
//    return distance::lorentzian(t1, t2);
//  }
//
//  F SBD::eval(const TSeries& t1, const TSeries& t2, F /* bsf */) {
//    return distance::sbd(t1, t2);
//  }

}
