#pragma once

#include "nn1splitters.hpp"
#include "MPGenerator.hpp"
#include "sp_basedist.hpp"

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>

namespace tempo::classifier::SForest::splitter::nn1 {

  /// 1NN Direct Alignment Distance
  struct DA : public BaseDist_i {

    double exponent;

    DA(std::string tname, double exponent) : BaseDist_i(std::move(tname)), exponent(exponent) {}

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override;
  };

  /// 1NN Direct Alignment Generator
  template<typename TrainS, typename TrainD>
  struct DAGen : public NN1SplitterDistanceGen<TrainS, TrainD> {
    using R = typename NN1SplitterDistanceGen<TrainS, TrainD>::R;

    TransformGetter<TrainS> get_transform;
    ExponentGetter<TrainS> get_exponent;

    DAGen(TransformGetter<TrainS> gt, ExponentGetter<TrainS> ge) :
      get_transform(std::move(gt)), get_exponent(std::move(ge)) {}

    R generate(std::unique_ptr<TrainS> state, const TrainD& /* data */, const ByClassMap& /* bcm */) override {
      // Generate args
      std::string tn = get_transform(*state);
      double e = get_exponent(*state);
      // Build return
      return {
        std::move(state),
        std::make_unique<DA>(tn, e)
      };
    }

  };

}
