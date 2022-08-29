#pragma once

#include "node_nn1dist.hpp"
#include "nodegen_nn1dist.hpp"
#include "MPGenerator.hpp"

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>

#include <tempo/distance/elastic/dtw.hpp>

namespace tempo::classifier::sf::node::nn1dist {

  struct DTWFull : public BaseDist {

    double exponent;

    DTWFull(std::string tname, double exponent) : BaseDist(std::move(tname)), exponent(exponent) {}

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override {
      constexpr auto d1 = distance::univariate::ad1<TSeries>;
      constexpr auto d2 = distance::univariate::ad2<TSeries>;
      if (exponent==1.0) { return distance::dtw(t1, t2, d1, utils::NO_WINDOW, bsf); }
      else if (exponent==2.0) { return distance::dtw(t1, t2, d2, utils::NO_WINDOW, bsf); }
      else { return distance::dtw(t1, t2, distance::univariate::ade<TSeries>(exponent), utils::NO_WINDOW, bsf); }
    }

    std::string get_distance_name() override { return "DTWFull:" + std::to_string(exponent); }
  };

  struct DTWFullGen : public i_GenDist {
    TransformGetter get_transform;
    ExponentGetter get_exponent;

    DTWFullGen(TransformGetter gt, ExponentGetter ge) :
      get_transform(std::move(gt)), get_exponent(std::move(ge)) {}

    std::unique_ptr<i_Dist> generate(TreeState& state, TreeData const& /*data*/, const ByClassMap& /* bcm */) override {
      // Generate args
      std::string tn = get_transform(state);
      double e = get_exponent(state);
      // Build return
      return std::make_unique<DTWFull>(tn, e);
    }
  };

} // End of namespace tempo::classifier::sf::node::nn1dist
