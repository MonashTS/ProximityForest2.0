#pragma once

#include "nn1splitters.hpp"
#include "MPGenerator.hpp"
#include "sp_basedist.hpp"

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>

#include <tempo/distance__/helpers.hpp>
#include <tempo/distance__/lockstep/direct.hpp>

namespace tempo::classifier::SForest::splitter::nn1 {

  /// 1NN ADTW Distance
  struct ADTW : public BaseDist_i {

    double exponent;
    F penalty;

    ADTW(std::string tname, double exponent, F penalty) :
      BaseDist_i(std::move(tname)), exponent(exponent), penalty(penalty) {}

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override;

    std::string get_distance_name() override {
      return "ADTW:" + std::to_string(exponent) + ":" + std::to_string(penalty);
    }
  };

  /// 1NN ADTW Generator
  template<typename TrainS, typename TrainD>
  struct ADTWGen : public NN1SplitterDistanceGen<TrainS, TrainD> {
    using R = typename NN1SplitterDistanceGen<TrainS, TrainD>::R;

    TransformGetter<TrainS> get_transform;
    ExponentGetter<TrainS> get_exponent;
    double omega_exponent = 5;

    ADTWGen(TransformGetter<TrainS> gt, ExponentGetter<TrainS> ge) :
      get_transform(std::move(gt)), get_exponent(std::move(ge)) {}

    R generate(std::unique_ptr<TrainS> state, TrainD const& data, ByClassMap const& bcm) override {
      // Generate args
      std::string tn = get_transform(*state);
      double e = get_exponent(*state);

      // Only sample if we haven't sample yet here
      // Cache in the state; warning: clear the cache on fork!
      if (!state->distance_splitter_state.ADTW_sampled_mean_da) {
        size_t n = bcm.size();
        size_t SAMPLE_SIZE = std::min<size_t>(4000, n*(n - 1)/2);
        DTS train_subset(data.get_train_dataset(tn), "subset", bcm.to_IndexSet());

        tempo::utils::StddevWelford welford;
        std::uniform_int_distribution<> distrib(0, (int)train_subset.size() - 1);
        for (size_t i = 0; i<SAMPLE_SIZE; ++i) {
          const auto& q = train_subset[distrib(state->prng)];
          const auto& s = train_subset[distrib(state->prng)];
          const auto& cost = distance::directa(q, s, tempo::distance::univariate::ade<TSeries>(e), tempo::utils::PINF);
          welford.update(cost);
        }
        state->distance_splitter_state.ADTW_sampled_mean_da = {welford.get_mean()};
      }

      // Generate penalty
      std::uniform_int_distribution<size_t> gen(0, 100); // uniform, unbiased
      size_t i = gen(state->prng);
      const double r = std::pow((double)i/100.0, omega_exponent);
      const double omega = r*state->distance_splitter_state.ADTW_sampled_mean_da.value();

      // Build return
      return {std::move(state), std::make_unique<ADTW>(tn, e, omega)};
    }

  };

}