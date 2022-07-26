#pragma once

#include "nn1splitters.hpp"
#include "MPGenerator.hpp"
#include "sp_basedist.hpp"

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>

namespace tempo::classifier::SForest::splitter::nn1 {

  /// 1NN ERP Distance
  struct ERP : public BaseDist_i {

    double exponent;
    double gv;
    size_t w;

    ERP(std::string tname, double exponent, double gv, size_t w) :
      BaseDist_i(std::move(tname)), exponent(exponent), gv(gv), w(w) {}

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override;

    std::string get_distance_name() override {
      return "ERP:" + std::to_string(exponent) + ":" + std::to_string(gv) + ":" + std::to_string(w);
    }
  };

  /// 1NN ERP Generator
  template<typename TrainS, typename TrainD>
  struct ERPGen : public NN1SplitterDistanceGen<TrainS, TrainD> {
    using R = typename NN1SplitterDistanceGen<TrainS, TrainD>::R;

    TransformGetter<TrainS> get_transform;
    ExponentGetter<TrainS> get_exponent;
    WindowGetter<TrainS, TrainD> get_window;
    StatGetter<TrainS, TrainD> get_gv;

    ERPGen(TransformGetter<TrainS> gt, ExponentGetter<TrainS> ge,
           WindowGetter<TrainS, TrainD> gw, StatGetter<TrainS, TrainD> gv) :
      get_transform(std::move(gt)), get_exponent(std::move(ge)), get_window(std::move(gw)), get_gv(std::move(gv)) {}

    R generate(std::unique_ptr<TrainS> state, const TrainD& data, const ByClassMap& bcm) override {
      // Generate args
      std::string tn = get_transform(*state);
      double e = get_exponent(*state);
      size_t w = get_window(*state, data);
      F gv = get_gv(*state, data, bcm, tn);
      // Build return
      return {
        std::move(state),
        std::make_unique<ERP>(tn, e, gv, w)
      };
    }

  };

}
