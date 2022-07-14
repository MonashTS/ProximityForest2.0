#pragma once

#include "nn1splitters.hpp"
#include "MPGenerator.hpp"
#include "sp_basedist.hpp"

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>

namespace tempo::classifier::SForest::splitter::nn1 {

  struct SBD : public BaseDist_i {

    explicit SBD(std::string tname) : BaseDist_i(std::move(tname)) {}

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override;
  };

  /// 1NN sbd Generator
  template<typename TrainS, typename TrainD>
  struct SBDGen : public NN1SplitterDistanceGen<TrainS, TrainD> {
    using R = typename NN1SplitterDistanceGen<TrainS, TrainD>::R;

    TransformGetter<TrainS> get_transform;

    explicit SBDGen(TransformGetter<TrainS> gt): get_transform(std::move(gt)) {}

    R generate(std::unique_ptr<TrainS> state, const TrainD& /* data */, const ByClassMap& /* bcm */) override {
      // Generate args
      std::string tn = get_transform(*state);
      // Build return
      return {
        std::move(state),
        std::make_unique<SBD>(tn)
      };
    }

  };

}
