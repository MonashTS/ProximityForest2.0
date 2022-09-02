#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>
#include <tempo/distance/univariate.hpp>

#include "nn1dist_base.hpp"
#include "tempo/distance/core/lockstep/direct.hpp"

namespace tempo::classifier::TSChief::snode::nn1splitter {

  struct ADTW : public BaseDist {

    double cfe;
    double penalty;

    ADTW(std::string tname, double cfe, double penalty) : BaseDist(std::move(tname)), cfe(cfe), penalty(penalty) {}

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override {
      return distance::univariate::adtw(t1.rawdata(), t1.size(), t2.rawdata(), t2.size(), cfe, penalty, bsf);
    }

    std::string get_distance_name() override { return "ADTW:" + std::to_string(cfe) + ":" + std::to_string(penalty); }
  };

  /// 1NN ADTW per node state
  struct ADTWGenState : public i_TreeState {

    std::optional<double> sample{std::nullopt};

    // --- --- --- Constructor/Destructor

    // --- --- --- Implement interface

    std::unique_ptr<i_TreeState> forest_fork(size_t) const override {
      return std::make_unique<ADTWGenState>();
    }

    void forest_merge_in(std::unique_ptr<i_TreeState>&&) override {}

    /// When starting a new branch, reset the sample
    void start_branch(size_t) override {
      sample = std::nullopt;
    }

    void end_branch(size_t) override {}

  };

  struct ADTWGen : public i_GenDist {

    static constexpr F omega_exponent = 5.0;

    TransformGetter get_transform;
    ExponentGetter get_fce;
    std::shared_ptr<i_GetState<ADTWGenState>> get_adtw_state;
    std::shared_ptr<i_GetData<std::map<std::string, DTS>>> get_train_data;

    ADTWGen(
      TransformGetter gt,
      ExponentGetter get_cfe,
      std::shared_ptr<i_GetState<ADTWGenState>> get_adtw_state,
      std::shared_ptr<i_GetData<std::map<std::string, DTS>>> get_train_data
    ) :
      get_transform(std::move(gt)),
      get_fce(std::move(get_cfe)),
      get_adtw_state(std::move(get_adtw_state)),
      get_train_data(std::move(get_train_data)) {}

    std::unique_ptr<i_Dist> generate(TreeState& state, TreeData const& data, const ByClassMap& bcm) override {
      const std::string tn = get_transform(state);
      const double e = get_fce(state);

      // --- Sampling
      // Only sample if we haven't sample at this node yet. Cache the result if we compute it.
      // Automatically cleared when starting a new branch.
      std::optional<double>& sample = get_adtw_state->at(state).sample;
      if (!sample) {
        // Create subset
        size_t n = bcm.size();
        size_t SAMPLE_SIZE = std::min<size_t>(4000, n*(n - 1)/2);
        DTS train_subset(get_train_data->at(data).at(tn), "subset", bcm.to_IndexSet());
        // Sample random pairs
        tempo::utils::StddevWelford welford;
        std::uniform_int_distribution<> distrib(0, (int)train_subset.size() - 1);
        for (size_t i = 0; i<SAMPLE_SIZE; ++i) {
          const auto& q = train_subset[distrib(state.prng)];
          const auto& s = train_subset[distrib(state.prng)];
          const F cost = distance::univariate::directa(q.rawdata(), q.size(), s.rawdata(), s.size(), e, utils::PINF);
          welford.update(cost);
        }
        sample = {welford.get_mean()};
      }

      // --- Compute penalty
      std::uniform_int_distribution<size_t> gen(0, 100); // uniform, unbiased
      size_t i = gen(state.prng);
      const double penalty = std::pow((double)i/100.0, omega_exponent)*sample.value();

      // Build return
      return std::make_unique<ADTW>(tn, e, penalty);
    }
  };

} // End of namespace tempo::classifier::TSChief::snode::nn1splitter
