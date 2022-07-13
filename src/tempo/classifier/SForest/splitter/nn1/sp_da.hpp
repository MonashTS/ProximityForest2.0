#pragma once

#include "nn1splitters.hpp"
#include "MPGenerator.hpp"

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>

namespace tempo::classifier::SForest::splitter::nn1 {

  /// 1NN Direct Alignment Distance
  struct DA : public Distance_i {

    std::string transformation_name;
    double exponent;

    DA(std::string tname, double exponent) : transformation_name(std::move(tname)), exponent(exponent) {}

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override;

    std::string get_transformation_name() override;
  };

  /// 1NN Direct Alignment Generator
  template<typename TrainState, typename TrainData>
  struct DAGen : public NN1SplitterDistanceGen<TrainState, TrainData> {
    using R = typename NN1SplitterDistanceGen<TrainState, TrainData>::R;

    TransformGetter<TrainState> get_transform;
    ExponentGetter<TrainState> get_exponent;

    DAGen(TransformGetter<TrainState> gt, ExponentGetter<TrainState> ge) :
      get_transform(std::move(gt)), get_exponent(std::move(ge)) {}

    R generate(std::unique_ptr<TrainState> state, const TrainData& /* data */, const ByClassMap& /* bcm */) override {
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
