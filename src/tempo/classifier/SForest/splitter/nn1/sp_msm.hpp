#pragma once

#include "nn1splitters.hpp"
#include "MPGenerator.hpp"
#include "sp_basedist.hpp"

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>

namespace tempo::classifier::SForest::splitter::nn1 {

  struct MSM : public BaseDist_i {

    double cost;

    MSM(std::string tname, double cost) :
      BaseDist_i(std::move(tname)), cost(cost) {}

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override;
  };

  /// 1NN MSM Generator
  template<typename TrainS, typename TrainD>
  struct MSMGen : public NN1SplitterDistanceGen<TrainS, TrainD> {
    using R = typename NN1SplitterDistanceGen<TrainS, TrainD>::R;
    using CostGetter = typename std::function<tempo::F(TrainS & state)>;

    TransformGetter <TrainS> get_transform;
    CostGetter get_cost;

    MSMGen(TransformGetter <TrainS> gt, CostGetter gc) :
      get_transform(std::move(gt)), get_cost(std::move(gc)) {}

    R generate(std::unique_ptr<TrainS> state, const TrainD& /* data */, const ByClassMap& /* bcm */) override {
      // Generate args
      std::string tn = get_transform(*state);
      double c = get_cost(*state);
      // Build return
      return {
        std::move(state),
        std::make_unique<MSM>(tn, c)
      };
    }

  };

}
