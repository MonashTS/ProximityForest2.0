#include "sp_da.hpp"
#include <tempo/distance/lockstep/direct.hpp>

namespace tempo::classifier::SForest::splitter::nn1 {

  F DA::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    return distance::directa(t1, t2, distance::univariate::ade<TSeries>(exponent), bsf);
  }

   std::string DA::get_transformation_name() {
    return transformation_name;
  }

}
