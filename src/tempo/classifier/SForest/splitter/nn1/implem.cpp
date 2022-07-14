#include "nn1splitters.hpp"

#include "sp_da.hpp"
#include <tempo/distance/lockstep/direct.hpp>

#include "sp_dtw.hpp"
#include <tempo/distance/elastic/dtw.hpp>

#include "sp_adtw.hpp"
#include <tempo/distance/elastic/adtw.hpp>

#include "sp_wdtw.hpp"
#include <tempo/distance/elastic/wdtw.hpp>

#include "sp_erp.hpp"
#include <tempo/distance/elastic/erp.hpp>

#include "sp_lcss.hpp"
#include <tempo/distance/elastic/lcss.hpp>

#include "sp_msm.hpp"
#include <tempo/distance/elastic/msm.hpp>

#include "sp_twe.hpp"
#include <tempo/distance/elastic/twe.hpp>

#include "sp_lorentzian.hpp"
#include <tempo/distance/lockstep/lockstep.hpp>

#include "sp_sbd.hpp"
#include <tempo/distance/sliding/cross_correlation.hpp>


namespace tempo::classifier::SForest::splitter::nn1 {

  F DA::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    return distance::directa(t1, t2, distance::univariate::ade<TSeries>(exponent), bsf);
  }

  F DTW::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    return distance::dtw(t1, t2, distance::univariate::ade<TSeries>(exponent), w, bsf);
  }

  F ADTW::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    return distance::adtw(t1, t2, distance::univariate::ade<TSeries>(exponent), penalty, bsf);
  }

  F WDTW::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    return distance::wdtw(t1, t2, distance::univariate::ade<TSeries>(exponent), weights, bsf);
  }

  F ERP::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    return distance::erp(t1, t2,
                         tempo::distance::univariate::adegv<TSeries>(exponent),
                         tempo::distance::univariate::ade<TSeries>(exponent),
                         w, gv, bsf);
  }

  F LCSS::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    return distance::lcss(t1, t2, distance::univariate::ade<TSeries>(exponent), w, epsilon, bsf);
  }

  F MSM::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    return distance::univariate::msm(t1, t2, cost, bsf);
  }

  F TWE::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    return distance::univariate::twe(t1, t2, nu, lambda, bsf);
  }

  F Lorentzian::eval(const TSeries& t1, const TSeries& t2, F /* bsf */) {
    return distance::lorentzian(t1, t2);
  }


  F SBD::eval(const TSeries& t1, const TSeries& t2, F /* bsf */) {
    return distance::sbd(t1, t2);
  }


}
