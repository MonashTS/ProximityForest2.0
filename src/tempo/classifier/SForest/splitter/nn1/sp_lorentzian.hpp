#pragma once

#include "nn1splitters.hpp"
#include "MPGenerator.hpp"
#include "sp_basedist.hpp"

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>

namespace tempo::classifier::SForest::splitter::nn1 {

  struct Lorentzian : public BaseDist_i {

    explicit Lorentzian(std::string tname) : BaseDist_i(std::move(tname)) {}

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override;

    std::string get_distance_name() override { return "Lorentzian"; }
  };

  /// 1NN Lorentzian Generator
  template<typename TrainS, typename TrainD>
  struct LorentzianGen : public NN1SplitterDistanceGen<TrainS, TrainD> {
    using R = typename NN1SplitterDistanceGen<TrainS, TrainD>::R;

    TransformGetter<TrainS> get_transform;

    explicit LorentzianGen(TransformGetter<TrainS> gt): get_transform(std::move(gt)) {}

    R generate(std::unique_ptr<TrainS> state, const TrainD& /* data */, const ByClassMap& /* bcm */) override {
      // Generate args
      std::string tn = get_transform(*state);
      // Build return
      return {
        std::move(state),
        std::make_unique<Lorentzian>(tn)
      };
    }

  };

}
