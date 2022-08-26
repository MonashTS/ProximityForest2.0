#pragma once

#include "node_nn1dist.hpp"
#include "nodegen_nn1dist.hpp"
#include "MPGenerator.hpp"

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>

#include <tempo/distance/lockstep/direct.hpp>

namespace tempo::classifier::sf::node::nn1dist {

  struct DA : public BaseDist {

    double exponent;

    DA(std::string tname, double exponent) : BaseDist(std::move(tname)), exponent(exponent) {}

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override {
      constexpr auto d1 = distance::univariate::ad1<TSeries>;
      constexpr auto d2 = distance::univariate::ad2<TSeries>;
      if (exponent==1.0) { return distance::directa(t1, t2, d1, bsf); }
      else if (exponent==2.0) { return distance::directa(t1, t2, d2, bsf); }
      else { return distance::directa(t1, t2, distance::univariate::ade<TSeries>(exponent), bsf); }
    }

    std::string get_distance_name() override { return "DA:" + std::to_string(exponent); }
  };

  struct DAGen : public i_GenDist {
    TransformGetter get_transform;
    ExponentGetter get_exponent;

    DAGen(TransformGetter gt, ExponentGetter ge) :
      get_transform(std::move(gt)), get_exponent(std::move(ge)) {}

    std::unique_ptr<i_Dist> generate(TreeState& state, TreeData const& /*data*/, const ByClassMap& /* bcm */) override {
      // Generate args
      std::string tn = get_transform(state);
      double e = get_exponent(state);
      // Build return
      return std::make_unique<DA>(tn, e);
    }
  };

} // End of namespace tempo::classifier::sf::node::nn1dist
