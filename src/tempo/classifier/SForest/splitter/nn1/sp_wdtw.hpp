#pragma once

#include "nn1splitters.hpp"
#include "MPGenerator.hpp"
#include "sp_basedist.hpp"

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>
#include <tempo/distance__/elastic/wdtw.hpp>

namespace tempo::classifier::SForest::splitter::nn1 {

  /// 1NN WDTW Distance
  struct WDTW : public BaseDist_i {

    double exponent;
    double g;
    std::vector<F> weights;

    WDTW(std::string tname, double exponent, double g, std::vector<F>&& weights) :
      BaseDist_i(std::move(tname)), exponent(exponent), g(g), weights(std::move(weights)) {}

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override;

    std::string get_distance_name() override { return "WDTW:" + std::to_string(exponent) + ":" + std::to_string(g); }

  };

  /// 1NN WDTW Generator
  template<typename TrainS, typename TrainD>
  struct WDTWGen : public NN1SplitterDistanceGen<TrainS, TrainD> {
    using R = typename NN1SplitterDistanceGen<TrainS, TrainD>::R;

    TransformGetter<TrainS> get_transform;
    ExponentGetter<TrainS> get_exponent;

    WDTWGen(TransformGetter<TrainS> gt, ExponentGetter<TrainS> ge) :
      get_transform(std::move(gt)), get_exponent(std::move(ge)) {}

    R generate(std::unique_ptr<TrainS> state, const TrainD& data, const ByClassMap& /* bcm */) override {
      // Generate args
      std::string tn = get_transform(*state);
      double e = get_exponent(*state);
      const F g = std::uniform_real_distribution<F>(0, 1)(state->prng);
      // Build return
      return {std::move(state),
              std::make_unique<WDTW>(tn, e, g, distance::generate_weights(g, data.get_train_header().length_max()))};
    }
  };

}