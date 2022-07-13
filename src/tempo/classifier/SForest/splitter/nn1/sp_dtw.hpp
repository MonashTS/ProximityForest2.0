#pragma once

#include "nn1splitters.hpp"
#include "MPGenerator.hpp"

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>

namespace tempo::classifier::SForest::splitter::nn1 {

  /// 1NN DTW Distance
  struct DTW : public Distance_i {

    std::string transformation_name;
    double exponent;
    size_t w;

    DTW(std::string tname, double exponent, size_t w) :
      transformation_name(std::move(tname)), exponent(exponent), w(w) {}

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override;

    std::string get_transformation_name() override;
  };

  /// 1NN DTW Generator
  template<typename TrainS, typename TrainD>
  struct DTWGen : public NN1SplitterDistanceGen<TrainS, TrainD> {
    using R = typename NN1SplitterDistanceGen<TrainS, TrainD>::R;

    TransformGetter<TrainS> get_transform;
    ExponentGetter<TrainS> get_exponent;
    WindowGetter<TrainS, TrainD> get_window;

    DTWGen(TransformGetter<TrainS> gt, ExponentGetter<TrainS> ge, WindowGetter<TrainS, TrainD> gw) :
      get_transform(std::move(gt)), get_exponent(std::move(ge)), get_window(std::move(gw)) {}

    R generate(std::unique_ptr<TrainS> state, const TrainD& data, const ByClassMap& /* bcm */) override {
      // Generate args
      std::string tn = get_transform(*state);
      double e = get_exponent(*state);
      size_t w = get_window(*state, data);
      // Build return
      return {std::move(state), std::make_unique<DTW>(tn, e, w)};
    }

  };

}