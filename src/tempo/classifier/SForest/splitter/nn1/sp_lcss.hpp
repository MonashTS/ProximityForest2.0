#pragma once

#include "nn1splitters.hpp"
#include "MPGenerator.hpp"
#include "sp_basedist.hpp"

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>

namespace tempo::classifier::SForest::splitter::nn1 {

  /// 1NN LCSS Distance
  struct LCSS : public BaseDist_i {

    double exponent;
    double epsilon;
    size_t w;

    LCSS(std::string tname, double exponent, double epsilon, size_t w) :
      BaseDist_i(std::move(tname)), exponent(exponent), epsilon(epsilon), w(w) {}

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override;

    std::string get_distance_name() override {
      return "LCSS:" + std::to_string(exponent) + ":" + std::to_string(w) + ":" + std::to_string(epsilon);
    }
  };

  /// 1NN Direct Alignment Generator
  template<typename TrainS, typename TrainD>
  struct LCSSGen : public NN1SplitterDistanceGen<TrainS, TrainD> {
    using R = typename NN1SplitterDistanceGen<TrainS, TrainD>::R;

    TransformGetter<TrainS> get_transform;
    ExponentGetter<TrainS> get_exponent;
    WindowGetter<TrainS, TrainD> get_window;
    StatGetter<TrainS, TrainD> get_epsilon;

    LCSSGen(TransformGetter<TrainS> gt, ExponentGetter<TrainS> gexponent,
            WindowGetter<TrainS, TrainD> gw, StatGetter<TrainS, TrainD> gepsilon) :
      get_transform(std::move(gt)), get_exponent(std::move(gexponent)),
      get_window(std::move(gw)), get_epsilon(std::move(gepsilon)) {}

    R generate(std::unique_ptr<TrainS> state, const TrainD& data, const ByClassMap& bcm) override {
      // Generate args
      std::string tn = get_transform(*state);
      double expo = get_exponent(*state);
      size_t w = get_window(*state, data);
      double epsi = get_epsilon(*state, data, bcm, tn);
      // Build return
      return {
        std::move(state),
        std::make_unique<LCSS>(tn, expo, epsi, w)
      };
    }

  };

}
