#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>
#include <tempo/distance/tseries.univariate.hpp>

#include "nn1dist_base.hpp"

namespace tempo::classifier::TSChief::snode::nn1splitter {

  struct ADTW : public BaseDist {
    F cfe;
    F penalty;

    ADTW(std::string tname, F cfe, F penalty) : BaseDist(std::move(tname)), cfe(cfe), penalty(penalty) {}

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override {
      return distance::univariate::adtw(t1, t2, cfe, penalty, bsf);
    }

    std::string get_distance_name() override { return "ADTW:" + std::to_string(cfe) + ":" + std::to_string(penalty); }
  };

  /// 1NN ADTW per node state
  struct ADTWGenState : public i_TreeState {
    // TODO: map : sample per transform!
    std::optional<F> sample{std::nullopt};

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
      const F e = get_fce(state);

      // --- Sampling
      // Only sample if we haven't sample at this node yet. Cache the result if we compute it.
      // Automatically cleared when starting a new branch.
      std::optional<F>& sample = get_adtw_state->at(state).sample;
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
          const F cost = distance::univariate::directa(q, s, e, utils::PINF);
          welford.update(cost);
        }
        // State updated here through mutable reference
        sample = {welford.get_mean()};
      }

      // --- Compute penalty
      const size_t i = std::uniform_int_distribution<size_t>(0, 100)(state.prng); // uniform, unbiased
      const F penalty = std::pow((F)i/100.0, omega_exponent)*sample.value();

      // Build return
      return std::make_unique<ADTW>(tn, e, penalty);
    }
  };

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // ADTWs1 : sample only once the max penalty
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  struct ADTWs1Gen : public i_GenDist {

    static constexpr F omega_exponent = 3.0;

    TransformGetter get_transform;
    ExponentGetter get_fce;
    std::map<std::tuple<F, std::string>, std::vector<F>> penalties;

    ADTWs1Gen(TransformGetter gt, ExponentGetter get_cfe,
              std::map<std::tuple<F, std::string>, std::vector<F>> penalties) :
      get_transform(std::move(gt)),
      get_fce(std::move(get_cfe)),
      penalties(penalties) {}

    std::unique_ptr<i_Dist> generate(TreeState& state, TreeData const& /*data*/ , const ByClassMap& /*bcm*/) override {
      const std::string tn = get_transform(state);
      const F e = get_fce(state);
      F penalty = utils::pick_one(penalties.at({e, tn}), state.prng);
      return std::make_unique<ADTW>(tn, e, penalty);
    }

    static std::map<std::tuple<F, std::string>, std::vector<F>> do_sampling(
      std::vector<F> const& exponent,
      std::vector<std::string> const& transforms,
      std::map<std::string, DTS> const& train_data,
      size_t SAMPLE_SIZE,
      PRNG& prng
    ) {

      // Number of penalty
      constexpr size_t NBP = 20;
      // Exponent when computing the penalties
      constexpr double OMEGA_EXPONENT = 5.0;

      std::map<std::tuple<F, std::string>, std::vector<F>> result;

      for (auto const& tn : transforms) {
        auto const& dts = train_data.at(tn);
        if (dts.size()<=1) { throw std::invalid_argument("DataSplit transform " + tn + " as less than 2 values"); }

        for (auto const& e : exponent) {
          // --- Sampling
          tempo::utils::StddevWelford welford;
          std::uniform_int_distribution<> distrib(0, (int)dts.size() - 1);
          for (size_t i = 0; i<SAMPLE_SIZE; ++i) {
            const auto& q = dts[distrib(prng)];
            const auto& s = dts[distrib(prng)];
            const F cost = distance::univariate::directa(q, s, e, utils::PINF);
            welford.update(cost);
          }
          F max_penalties = welford.get_mean();
          // --- Compute penalties
          std::vector<F> penalties;
          penalties.push_back(0.0);
          for (size_t i = 1; i<NBP; ++i) {
            const F penalty = std::pow((F)i/20.0, OMEGA_EXPONENT)*max_penalties;
            penalties.push_back(penalty);
          }

          result[std::tuple(e, tn)] = penalties;
        }
      }

      return result;
    }

  };

} // End of namespace tempo::classifier::TSChief::snode::nn1splitter
