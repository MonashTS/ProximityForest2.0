#pragma once

#include "nn1splitters.hpp"
#include "MPGenerator.hpp"
#include "sp_basedist.hpp"

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>

namespace tempo::classifier::SForest::splitter::nn1 {

  struct TWE : public BaseDist_i {

    double nu;
    double lambda;

    TWE(std::string tname, double nu, double lambda) :
      BaseDist_i(std::move(tname)), nu(nu), lambda(lambda) {}

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override;
  };

  /// 1NN TWE Generator
  template<typename TrainS, typename TrainD>
  struct TWEGen : public NN1SplitterDistanceGen<TrainS, TrainD> {
    using R = typename NN1SplitterDistanceGen<TrainS, TrainD>::R;
    using Getter = typename std::function<F(TrainS& state)>;

    TransformGetter <TrainS> get_transform;
    Getter get_nu;
    Getter get_lambda;

    TWEGen(TransformGetter <TrainS> gt, Getter gnu, Getter glambda) :
      get_transform(std::move(gt)), get_nu(std::move(gnu)), get_lambda(std::move(glambda)) {}

    R generate(std::unique_ptr<TrainS> state, const TrainD& /* data */, const ByClassMap& /* bcm */) override {
      // Generate args
      std::string tn = get_transform(*state);
      double nu = get_nu(*state);
      double lambda = get_lambda(*state);
      // Build return
      return {
        std::move(state),
        std::make_unique<TWE>(tn, nu, lambda)
      };
    }

  };

}
