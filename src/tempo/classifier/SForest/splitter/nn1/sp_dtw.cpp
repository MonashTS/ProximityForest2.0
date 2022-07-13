#include "sp_dtw.hpp"
#include <tempo/distance/elastic/dtw.hpp>

namespace tempo::classifier::SForest::splitter::nn1 {

  F DTW::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    return distance::dtw(t1, t2, distance::univariate::ade<TSeries>(exponent), w, bsf);
  }

  std::string DTW::get_transformation_name() {
    return transformation_name;
  }

}
