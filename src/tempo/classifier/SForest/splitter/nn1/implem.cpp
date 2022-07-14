#include "nn1splitters.hpp"

#include "sp_da.hpp"
#include <tempo/distance/lockstep/direct.hpp>

#include "sp_dtw.hpp"
#include <tempo/distance/elastic/dtw.hpp>

#include "sp_adtw.hpp"
#include <tempo/distance/elastic/adtw.hpp>

#include "sp_wdtw.hpp"
#include <tempo/distance/elastic/wdtw.hpp>

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

}
