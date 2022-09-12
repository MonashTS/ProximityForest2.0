#include "pf2018.hpp"

#include "tempo/classifier/TSChief/tree.hpp"
#include "tempo/classifier/TSChief/forest.hpp"
#include "tempo/classifier/TSChief/sleaf/pure_leaf.hpp"
#include "tempo/classifier/TSChief/snode/meta/chooser.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1splitter.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1_directa.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1_adtw.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1_dtw.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1_dtwfull.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1_wdtw.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1_erp.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1_lcss.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1_msm.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1_twe.hpp"

namespace pf2018::splitters {

  using F = double;
  using MDTS = std::map<std::string, tempo::DTS>;
  namespace tsc = tempo::classifier::TSChief;
  namespace tsc_nn1 = tempo::classifier::TSChief::snode::nn1splitter;

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Parameterization
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  // --- --- --- Cost function exponent

  tsc_nn1::ExponentGetter make_get_vcfe(std::vector<F> exponent_set) {
    return [es = std::move(exponent_set)](tsc::TreeState& s) { return tempo::utils::pick_one(es, s.prng); };
  }

  tsc_nn1::ExponentGetter make_get_cfe1() { return [](tsc::TreeState& /*s*/) { return 1.0; }; }

  tsc_nn1::ExponentGetter make_get_cfe2() { return [](tsc::TreeState& /*s*/) { return 2.0; }; }

  // --- --- --- Transform getters

  tsc_nn1::TransformGetter make_get_transform(std::vector<std::string> tr_set) {
    return [ts = std::move(tr_set)](tsc::TreeState& s) { return tempo::utils::pick_one(ts, s.prng); };
  }

  tsc_nn1::TransformGetter make_get_derivative(size_t d) {
    return [d](tsc::TreeState& /*s*/) { return "derivative" + std::to_string(d); };
  }

  // --- --- --- Window getter

  tsc_nn1::WindowGetter make_get_window(size_t maxlength) {
    return [=](tsc::TreeState& s, tsc::TreeData const& /* d */) {
      const size_t win_top = std::floor((double)maxlength + 1/4.0);
      return std::uniform_int_distribution<size_t>(0, win_top)(s.prng);
    };
  }

  // --- --- --- ERP Gap Value *AND* LCSS epsilon.

  tsc_nn1::StatGetter make_get_frac_stddev(const std::shared_ptr<tsc::i_GetData<MDTS>>& get_train_data) {
    return [=](tsc::TreeState& s, tsc::TreeData const& data, tempo::ByClassMap const& bcm, std::string const& tr_name) {
      const tempo::DTS& train_dataset = get_train_data->at(data).at(tr_name);
      auto stddev_ = stddev(train_dataset, bcm.to_IndexSet());
      return std::uniform_real_distribution<F>(stddev_/5.0, stddev_)(s.prng);
    };
  }

  // --- --- --- MSM Cost

  tsc_nn1::T_GetterState<F> make_get_msm_cost() {
    return [](tsc::TreeState& state) {
      constexpr size_t MSM_N = 100;
      constexpr F msm_cost[MSM_N]{
        0.01, 0.01375, 0.0175, 0.02125, 0.025, 0.02875, 0.0325, 0.03625, 0.04, 0.04375,
        0.0475, 0.05125, 0.055, 0.05875, 0.0625, 0.06625, 0.07, 0.07375, 0.0775, 0.08125,
        0.085, 0.08875, 0.0925, 0.09625, 0.1, 0.136, 0.172, 0.208, 0.244, 0.28, 0.316, 0.352,
        0.388, 0.424, 0.46, 0.496, 0.532, 0.568, 0.604, 0.64, 0.676, 0.712, 0.748, 0.784,
        0.82, 0.856, 0.892, 0.928, 0.964, 1, 1.36, 1.72, 2.08, 2.44, 2.8, 3.16, 3.52, 3.88,
        4.24, 4.6, 4.96, 5.32, 5.68, 6.04, 6.4, 6.76, 7.12, 7.48, 7.84, 8.2, 8.56, 8.92, 9.28,
        9.64, 10, 13.6, 17.2, 20.8, 24.4, 28, 31.6, 35.2, 38.8, 42.4, 46, 49.6, 53.2, 56.8,
        60.4, 64, 67.6, 71.2, 74.8, 78.4, 82, 85.6, 89.2, 92.8, 96.4, 100
      };
      return tempo::utils::pick_one(msm_cost, MSM_N, state.prng);
    };
  }

  // --- --- --- TWE nu & lambda parameters

  tsc_nn1::T_GetterState<F> make_get_twe_nu() {
    return [](tsc::TreeState& state) {
      constexpr size_t N = 10;
      constexpr F nus[N]{0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1};
      return tempo::utils::pick_one(nus, N, state.prng);
    };
  }

  tsc_nn1::T_GetterState<F> make_get_twe_lambda() {
    return [](tsc::TreeState& state) {
      constexpr size_t N = 10;
      constexpr F lambdas[N]{0, 0.011111111, 0.022222222, 0.033333333, 0.044444444,
                             0.055555556, 0.066666667, 0.077777778, 0.088888889, 0.1};
      return tempo::utils::pick_one(lambdas, N, state.prng);
    };
  }


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Splitters
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---


  // --- --- --- Leaf Generator

  std::shared_ptr<tsc::i_GenLeaf> make_pure_leaf(
    std::shared_ptr<tsc::i_GetData<tempo::DatasetHeader>> const& get_train_header
  ) {
    return std::make_shared<tsc::sleaf::GenLeaf_Pure>(get_train_header);
  }

  std::shared_ptr<tsc::i_GenNode> make_node_splitter(
    std::vector<F> exponents,
    std::vector<std::string> transforms,
    std::set<std::string> const& distances,
    size_t nbc,
    size_t series_max_length,
    std::shared_ptr<tsc::i_GetData<std::map<std::string, tempo::DTS>>> const& get_train_data,
    std::shared_ptr<tsc::i_GetData<std::map<std::string, tempo::DTS>>> const& get_test_data,
    tsc::TreeState& tstate
  ) {

    // --- --- --- State
    // 1NN distance splitters - cache the indexset
    using GS1NNState = tsc_nn1::GenSplitterNN1_State;
    std::shared_ptr<tsc::i_GetState<GS1NNState>> get_GenSplitterNN1_State =
      tstate.register_state<GS1NNState>(std::make_unique<GS1NNState>());

    // --- --- --- Getters
    auto getter_cfe_set = make_get_vcfe(std::move(exponents));
    auto getter_cfe_2 = make_get_cfe2();
    auto getter_tr_set = make_get_transform(std::move(transforms));
    auto getter_window = make_get_window(series_max_length);
    auto frac_stddev = make_get_frac_stddev(get_train_data);

    // --- --- --- Build distance generators

    // List of distance generators (this is specific to our collection of NN1 Splitter generators)
    std::vector<std::shared_ptr<tsc_nn1::i_GenDist>> gendist;

    for (std::string const& sname : distances) {

      if (sname.starts_with("DA")) {
        // --- --- --- Direct Alignment
        gendist.push_back(make_shared<tsc_nn1::DAGen>(getter_tr_set, getter_cfe_set));
      } else if (sname.starts_with("ADTW")) {
        // --- --- --- ADTW
        // ADTW, with state and access to train data for sampling
        // ADTW train - cache sampling
        std::shared_ptr<tsc::i_GetState<tsc_nn1::ADTWGenState>> get_adtw_state =
          tstate.register_state<tsc_nn1::ADTWGenState>(std::make_unique<tsc_nn1::ADTWGenState>());
        //
        gendist.push_back(make_shared<tsc_nn1::ADTWGen>(getter_tr_set, getter_cfe_set, get_adtw_state, get_train_data));
      } else if (sname.starts_with("DTW")&&!sname.starts_with("DTWFull")) {
        // --- --- --- DTW
        gendist.push_back(make_shared<tsc_nn1::DTWGen>(getter_tr_set, getter_cfe_set, getter_window));
      } else if (sname.starts_with("WDTW")) {
        // --- --- --- WDTW
        gendist.push_back(make_shared<tsc_nn1::WDTWGen>(getter_tr_set, getter_cfe_set, series_max_length));
      } else if (sname.starts_with("DTWFull")) {
        // --- --- --- DTWFull
        gendist.push_back(make_shared<tsc_nn1::DTWFullGen>(getter_tr_set, getter_cfe_set));
      } else if (sname.starts_with("ERP")) {
        // --- --- --- ERP
        gendist.push_back(make_shared<tsc_nn1::ERPGen>(getter_tr_set, getter_cfe_2, frac_stddev, getter_window));
      } else if (sname.starts_with("LCSS")) {
        // --- --- --- LCSS
        gendist.push_back(make_shared<tsc_nn1::LCSSGen>(getter_tr_set, frac_stddev, getter_window));
      } else if (sname.starts_with("MSM")) {
        // --- --- --- MSM
        auto getter_msm_cost = make_get_msm_cost();
        gendist.push_back(make_shared<tsc_nn1::MSMGen>(getter_tr_set, getter_msm_cost));
      } else if (sname.starts_with("TWE")) {
        // --- --- --- TWE
        auto getter_twe_nu = make_get_twe_nu();
        auto getter_twe_lambda = make_get_twe_lambda();
        gendist.push_back(make_shared<tsc_nn1::TWEGen>(getter_tr_set, getter_twe_nu, getter_twe_lambda));
      }
    }

    // Build vector for the node generator
    std::vector<std::shared_ptr<tsc::i_GenNode>> generators;

    // Wrap each distance generator in GenSplitter1NN (which is a i_GenNode) and push in generators
    for (auto const& gd : gendist) {
      generators.push_back(
        make_shared<tsc_nn1::GenSplitterNN1>(gd, get_GenSplitterNN1_State, get_train_data, get_test_data));
    }

    // --- Put a node chooser over all generators
    return make_shared<tsc::snode::meta::SplitterChooserGen>(std::move(generators), nbc);
  }

} // End of namespace pf2018::splitters