#include "pf2018.hpp"

#include <tempo/classifier/SForest/sforest.hpp>
#include <tempo/classifier/SForest/splitter/meta/chooser.hpp>
#include <tempo/classifier/SForest/leaf/pure_leaf.hpp>

#include <tempo/classifier/SForest/splitter/nn1/nn1splitters.hpp>
#include <tempo/classifier/SForest/splitter/nn1/MPGenerator.hpp>
#include <tempo/classifier/SForest/splitter/nn1/sp_da.hpp>
#include <tempo/classifier/SForest/splitter/nn1/sp_adtw.hpp>
#include <tempo/classifier/SForest/splitter/nn1/sp_dtw.hpp>
#include <tempo/classifier/SForest/splitter/nn1/sp_wdtw.hpp>
#include <tempo/classifier/SForest/splitter/nn1/sp_erp.hpp>
#include <tempo/classifier/SForest/splitter/nn1/sp_lcss.hpp>
#include <tempo/classifier/SForest/splitter/nn1/sp_lorentzian.hpp>
#include <tempo/classifier/SForest/splitter/nn1/sp_msm.hpp>
#include <tempo/classifier/SForest/splitter/nn1/sp_sbd.hpp>
#include <tempo/classifier/SForest/splitter/nn1/sp_twe.hpp>
#include <tempo/distance/helpers.hpp>

#include <tempo/transform/derivative.hpp>
#include <utility>

namespace tempo::classifier {

  namespace {
    using namespace std;
    using namespace tempo::classifier::SForest::splitter::nn1;

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // State and Data structures
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Note: we use the same structures for both train and test time.
    // Some fields may be specific to one phase or the other.

    struct state {
      size_t seed;
      PRNG prng;

      SForest::splitter::nn1::NN1SplitterState distance_splitter_state;

      explicit state(size_t seed) : seed(seed), prng(seed) {}

      explicit state(PRNG&& prng) : prng(prng) {}

      state branch_fork(size_t /* branch_idx */) { return state(move(prng)); }

      void branch_merge(state&& s) { prng = move(s.prng); }

      state forest_fork(size_t branch_idx) { return state(seed + branch_idx); }

      void forest_merge(state&& /* s */ ) {}

    };
    static_assert(SForest::TreeState<state>);
    static_assert(SForest::ForestState<state>);
    static_assert(SForest::splitter::nn1::HasNN1SplitterState<state>);

    struct data {

      map<string, DTS> trainset{};  /// For both train and test time
      map<string, DTS> testset{};   /// Only at test time

      inline explicit data(map<string, DTS> trainset) : trainset(move(trainset)) {}

      inline data(map<string, DTS> trainset, map<string, DTS> testset) :
        trainset(move(trainset)),
        testset(move(testset)) {}

      /// Concept TrainData: access to the training set
      inline DTS get_train_dataset(const std::string& tname) const { return trainset.at(tname); }

      /// Concept TrainData: access to the training header
      inline DatasetHeader const& get_train_header() const { return trainset.begin()->second.header(); }

      /// NN1TestData concepts requirement
      inline DTS get_test_dataset(const std::string& tname) const { return testset.at(tname); }

      /// Concept TestData: access to the training header
      inline DatasetHeader const& get_test_header() const { return testset.begin()->second.header(); }

    };
    static_assert(SForest::TrainData<data>);
    static_assert(SForest::TestData<data>);

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Transforms

    /// Possible transforms
    //vector<string> transforms{"default", "derivative1", "derivative2"};
    vector<string> transforms{"default", "derivative1"};

    /// Pick transform among the possible one
    TransformGetter<state> transform_getter = [](state& state) -> string {
      return utils::pick_one(transforms, state.prng);
    };

    /// Pick default transform
    TransformGetter<state> transform_getter_default = [](state& /* state */) -> string {
      return "default";
    };

    /// Pick derivative 1 transform
    TransformGetter<state> transform_getter_derivative1 = [](state& /* state */) -> string {
      return "derivative1";
    };

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Exponents

    vector<double> exponents{0.5, 0.67, 1.0, 1.5, 2.0};

    /// Pick Exponent
    ExponentGetter<state> exp_getter = [](state& state) -> double { return utils::pick_one(exponents, state.prng); };

    /// Pick Exponent always at 2.0
    ExponentGetter<state> exp_2 = [](state& /* state */) -> double { return 2.0; };

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Others

    /// Random window computation function [0, Lmax/4]
    WindowGetter<state, data> window_getter = [](state& state, data const& data) -> size_t {
      const size_t win_top = std::floor((double)data.get_train_header().length_max() + 1/4.0);
      return std::uniform_int_distribution<size_t>(0, win_top)(state.prng);
    };

    /// ERP Gap Value *AND* LCSS epsilon.
    /// Random fraction of the dataset standard deviation, within [stddev/5, stddev[
    StatGetter<state, data> frac_stddev =
      [](state& state, data const& data, ByClassMap const& bcm, string const& transform_name) -> double {
        const auto& train_dataset = data.get_train_dataset(transform_name);
        auto stddev_ = stddev(train_dataset, bcm.to_IndexSet());
        return std::uniform_real_distribution<F>(stddev_/5.0, stddev_)(state.prng);
      };

    /// MSM costs
    MSMGen<state, data>::CostGetter msm_cost = [](state& state) -> double {
      constexpr size_t N = 100;
      double costs[N]{
        0.01, 0.01375, 0.0175, 0.02125, 0.025, 0.02875, 0.0325, 0.03625, 0.04, 0.04375,
        0.0475, 0.05125, 0.055, 0.05875, 0.0625, 0.06625, 0.07, 0.07375, 0.0775, 0.08125,
        0.085, 0.08875, 0.0925, 0.09625, 0.1, 0.136, 0.172, 0.208, 0.244, 0.28, 0.316, 0.352,
        0.388, 0.424, 0.46, 0.496, 0.532, 0.568, 0.604, 0.64, 0.676, 0.712, 0.748, 0.784,
        0.82, 0.856, 0.892, 0.928, 0.964, 1, 1.36, 1.72, 2.08, 2.44, 2.8, 3.16, 3.52, 3.88,
        4.24, 4.6, 4.96, 5.32, 5.68, 6.04, 6.4, 6.76, 7.12, 7.48, 7.84, 8.2, 8.56, 8.92, 9.28,
        9.64, 10, 13.6, 17.2, 20.8, 24.4, 28, 31.6, 35.2, 38.8, 42.4, 46, 49.6, 53.2, 56.8,
        60.4, 64, 67.6, 71.2, 74.8, 78.4, 82, 85.6, 89.2, 92.8, 96.4, 100
      };
      return utils::pick_one(costs, N, state.prng);
    };

    /// TWE nu parameters
    TWEGen<state, data>::Getter twe_nu = [](state& state) -> double {
      constexpr size_t N = 10;
      double nus[N]{0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1};
      return utils::pick_one(nus, N, state.prng);
    };

    /// TWE lambda parameters
    TWEGen<state, data>::Getter twe_lambda = [](state& state) -> double {
      constexpr size_t N = 10;
      double lambdas[N]{0, 0.011111111, 0.022222222, 0.033333333, 0.044444444,
                        0.055555556, 0.066666667, 0.077777778, 0.088888889, 0.1};
      return utils::pick_one(lambdas, N, state.prng);
    };

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // PF2018_11

    /// Generate the node splitter for the  2018.11 version of PF, for nbc candidate per nodes
    std::shared_ptr<SForest::NodeSplitterGen_i<state, data, state, data>> get_node_gen_11(size_t nbc) {

      // Direct Alignment default
      auto nn1da_def_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<DAGen<state, data>>(transform_getter_default, exp_2)
      );

      // DTW default
      auto nn1dtw_def_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<DTWGen<state, data>>(transform_getter_default, exp_2, window_getter)
      );

      // DTW derivative1
      auto nn1dtw_d1_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<DTWGen<state, data>>(transform_getter_derivative1, exp_2, window_getter)
      );

      // DTWfull default
      auto nn1dtwfull_def_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<DTWfullGen<state, data>>(transform_getter_default, exp_2)
      );

      // DTWfull derivative1
      auto nn1dtwfull_d1_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<DTWfullGen<state, data>>(transform_getter_derivative1, exp_2)
      );

      // WDTW default
      auto nn1wdtw_def_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<WDTWGen<state, data>>(transform_getter_default, exp_2)
      );

      // WDTW derivative1
      auto nn1wdtw_d1_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<WDTWGen<state, data>>(transform_getter_derivative1, exp_2)
      );

      // ERP
      auto nn1erp_def_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<ERPGen<state, data>>(transform_getter_default, exp_2, window_getter, frac_stddev)
      );

      // LCSS
      auto nn1lcss_def_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<LCSSGen<state, data>>(transform_getter_default, exp_2, window_getter, frac_stddev)
      );

      // MSM
      auto nn1msm_def_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<MSMGen<state, data>>(transform_getter_default, msm_cost)
      );

      // TWE
      auto nn1twe_def_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<TWEGen<state, data>>(transform_getter_default, twe_nu, twe_lambda)
      );

      return make_shared<SForest::splitter::meta::SplitterChooserGen<state, data, state, data>>(
        vector<shared_ptr<SForest::NodeSplitterGen_i<state, data, state, data>>>{
          nn1da_def_gen,
          nn1dtw_def_gen,
          nn1dtw_d1_gen,
          nn1dtwfull_def_gen,
          nn1dtwfull_d1_gen,
          nn1wdtw_def_gen,
          nn1wdtw_d1_gen,
          nn1erp_def_gen,
          nn1lcss_def_gen,
          nn1msm_def_gen,
          nn1twe_def_gen
        },
        nbc
      );

    } // End of get_node_gen_11


    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // PF2018_11_vcfe with Variable Cost Function Exponent

    /// Generate the node splitter for the  2018_11_vcfe version of PF, for nbc candidate per nodes
    std::shared_ptr<SForest::NodeSplitterGen_i<state, data, state, data>> get_node_gen_11_vcfe(size_t nbc) {

      // Direct Alignment default
      auto nn1da_def_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<DAGen<state, data>>(transform_getter_default, exp_getter)
      );

      // DTW default
      auto nn1dtw_def_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<DTWGen<state, data>>(transform_getter_default, exp_getter, window_getter)
      );

      // DTW derivative1
      auto nn1dtw_d1_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<DTWGen<state, data>>(transform_getter_derivative1, exp_getter, window_getter)
      );

      // DTWfull default
      auto nn1dtwfull_def_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<DTWfullGen<state, data>>(transform_getter_default, exp_getter)
      );

      // DTWfull derivative1
      auto nn1dtwfull_d1_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<DTWfullGen<state, data>>(transform_getter_derivative1, exp_getter)
      );

      // WDTW default
      auto nn1wdtw_def_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<WDTWGen<state, data>>(transform_getter_default, exp_getter)
      );

      // WDTW derivative1
      auto nn1wdtw_d1_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<WDTWGen<state, data>>(transform_getter_derivative1, exp_getter)
      );

      // ERP
      auto nn1erp_def_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<ERPGen<state, data>>(transform_getter_default, exp_2, window_getter, frac_stddev)
      );

      // LCSS
      auto nn1lcss_def_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<LCSSGen<state, data>>(transform_getter_default, exp_2, window_getter, frac_stddev)
      );

      // MSM
      auto nn1msm_def_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<MSMGen<state, data>>(transform_getter_default, msm_cost)
      );

      // TWE
      auto nn1twe_def_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<TWEGen<state, data>>(transform_getter_default, twe_nu, twe_lambda)
      );

      return make_shared<SForest::splitter::meta::SplitterChooserGen<state, data, state, data>>(
        vector<shared_ptr<SForest::NodeSplitterGen_i<state, data, state, data>>>{
          nn1da_def_gen,
          nn1dtw_def_gen,
          nn1dtw_d1_gen,
          nn1dtwfull_def_gen,
          nn1dtwfull_d1_gen,
          nn1wdtw_def_gen,
          nn1wdtw_d1_gen,
          nn1erp_def_gen,
          nn1lcss_def_gen,
          nn1msm_def_gen,
          nn1twe_def_gen
        },
        nbc
      );

    } // End of get_node_gen_11_vcfe


    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // PF2018_22

    /// Generate the node splitter for the  2018_22 version of PF, for nbc candidate per nodes
    std::shared_ptr<SForest::NodeSplitterGen_i<state, data, state, data>> get_node_gen_22(size_t nbc) {

      // Direct Alignment
      auto nn1da_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<DAGen<state, data>>(transform_getter, exp_getter)
      );

      // DTW
      auto nn1dtw_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<DTWGen<state, data>>(transform_getter, exp_getter, window_getter)
      );

      // DTWfull
      auto nn1dtwfull_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<DTWfullGen<state, data>>(transform_getter, exp_getter)
      );

      // ADTW
      auto nn1adtw_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<ADTWGen<state, data>>(transform_getter, exp_getter)
      );

      // WDTW
      auto nn1wdtw_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<WDTWGen<state, data>>(transform_getter, exp_getter)
      );

      // ERP
      auto nn1erp_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<ERPGen<state, data>>(transform_getter, exp_2, window_getter, frac_stddev)
      );

      // LCSS
      auto nn1lcss_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<LCSSGen<state, data>>(transform_getter, exp_2, window_getter, frac_stddev)
      );

      // Lorentzian
      auto nn1lorentzian_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<LorentzianGen<state, data>>(transform_getter)
      );

      // MSM
      auto nn1msm_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<MSMGen<state, data>>(transform_getter, msm_cost)
      );

      // SBD
      auto nn1sbd_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<SBDGen<state, data>>(transform_getter)
      );

      // TWE
      auto nn1twe_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<TWEGen<state, data>>(transform_getter, twe_nu, twe_lambda)
      );

      return make_shared<SForest::splitter::meta::SplitterChooserGen<state, data, state, data>>(
        vector<shared_ptr<SForest::NodeSplitterGen_i<state, data, state, data>>>{
          nn1da_gen,
          nn1dtw_gen,
          nn1dtwfull_gen,
          nn1adtw_gen,
          nn1wdtw_gen,
          nn1erp_gen,
          nn1lcss_gen,
          nn1lorentzian_gen,
          nn1msm_gen,
          nn1sbd_gen,
          nn1twe_gen
        },
        nbc
      );
    } // End of get_node_gen_22



    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // PF2018_ADTW_LCSS

    /// Generate the node splitter for the  2018.11 version of PF, for nbc candidate per nodes
    std::shared_ptr<SForest::NodeSplitterGen_i<state, data, state, data>> get_node_gen_adtw_lcss(size_t nbc) {

      // ADTW
      auto nn1adtw_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<ADTWGen<state, data>>(transform_getter, exp_getter)
      );

      // LCSS
      auto nn1lcss_gen = make_shared<NN1SplitterGen<state, data, state, data>>(
        make_shared<LCSSGen<state, data>>(transform_getter, exp_2, window_getter, frac_stddev)
      );

      return make_shared<SForest::splitter::meta::SplitterChooserGen<state, data, state, data>>(
        vector<shared_ptr<SForest::NodeSplitterGen_i<state, data, state, data>>>{
          nn1adtw_gen,
          nn1lcss_gen,
        },
        nbc
      );
    } // End of get_node_gen_adtw_lcss




  } // End of anonymous namespace


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // PF2018 IMPLEMENTATION
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  struct PF2018::Impl {

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Fields
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    // --- --- --- Valid after construction
    size_t nb_trees{};
    size_t nb_candidates{};
    std::string pfversion{};

    // --- --- --- Valid after train
    std::optional<DTSMap> o_trainset{};    // Train set needed at test time (for distance to exemplar)
    std::unique_ptr<SForest::SForest<state, data>> trained_forest{};


    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Construction
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    Impl(size_t nb_trees, size_t nb_candidates, std::string pfversion)
      : nb_trees(nb_trees), nb_candidates(nb_candidates), pfversion(std::move(pfversion)) {}


    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Functions
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Train the classifier
    void train(std::map<std::string, DTS> train_set_map, size_t seed, size_t nb_threads) {

      // --- --- --- Arg check

      if (!train_set_map.contains("default")) {
        throw std::invalid_argument("'default' transform not found in trainset");
      }
      if (!train_set_map.contains("derivative1")) {
        throw std::invalid_argument("'derivative1' transform not found in trainset");
      }
      if (!valid_versions.contains(pfversion)) {
        throw std::invalid_argument("Unrecognized pfversion");
      }

      // --- --- --- Build the train 'state' and 'data'

      auto train_state = std::make_unique<state>(seed);
      data train_data(train_set_map);

      DTS const& train_dataset = train_set_map.at("default");
      //    ByClassMap  vector<size_t>
      auto [train_bcm, train_bcm_remains] = train_dataset.get_BCM();
      if (!train_bcm_remains.empty()) { throw std::invalid_argument("train instances without label"); }

      // --- --- --- Build the node and leaf generators

      std::shared_ptr<SForest::NodeSplitterGen_i<state, data, state, data>> splitter_gen;
      if (pfversion==pf2018_11) {
        splitter_gen = get_node_gen_11(nb_candidates);
      } else if (pfversion==pf2018_11_vcfe) {
        splitter_gen = get_node_gen_11_vcfe(nb_candidates);
      } else if (pfversion==pf2018_22) {
        splitter_gen = get_node_gen_22(nb_candidates);
      } else if (pfversion==pf2018_adtw_lcss) {
        splitter_gen = get_node_gen_adtw_lcss(nb_candidates);
      }

      auto pleaf_gen = make_shared<SForest::leaf::PureLeaf_Gen<state, data, state, data>>();

      // --- --- --- Train

      auto tree_trainer = std::make_shared<SForest::STreeTrainer<state, data, state, data>>(pleaf_gen, splitter_gen);
      SForest::SForestTrainer<state, data, state, data> forest_trainer(tree_trainer, nb_trees);

      auto [res_state1, res_forest] = forest_trainer.train(move(train_state), train_data, train_bcm, nb_threads);

      // --- --- --- Store results
      o_trainset = {std::move(train_set_map)};
      trained_forest = std::move(res_forest);
    }

    /// Predict probabilities for the testset. Must be called after 'train'!
    classifier::ResultN predict(const DTSMap& test_set_map, size_t seed, size_t nb_threads) {

      // --- --- --- Check calling logic

      if (!o_trainset) { throw std::logic_error("'predict' called before 'train'"); }
      const auto& train_set_map = o_trainset.value();

      // --- --- --- Arg check

      if (!test_set_map.contains("default")) {
        throw std::invalid_argument("'default' transform not found in trainset");
      }
      if (!test_set_map.contains("derivative1")) {
        throw std::invalid_argument("'derivative1' transform not found in trainset");
      }

      // --- --- --- Build the train 'state' and 'data'

      auto test_state = std::make_unique<state>(seed);
      data train_test_data(train_set_map, test_set_map);

      // --- --- --- Test
      ResultN result;

      const size_t test_size = test_set_map.at("default").size();

      for (size_t test_idx = 0; test_idx<test_size; ++test_idx) {
        auto [test_state1, res1] =
          trained_forest->predict(std::move(test_state), train_test_data, test_idx, nb_threads);
        test_state = std::move(test_state1);  // Transmit state
        result.append(res1);
      }

      return result;
    }

  }; // End of struct PF2018::Impl

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // PF2018 INTERFACE
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  PF2018::PF2018(size_t nb_trees, size_t nb_candidates, std::string pfversion)
    : pImpl(std::make_unique<Impl>(nb_trees, nb_candidates, std::move(pfversion))) {}

  PF2018::~PF2018() = default;

  void PF2018::train(DTSMap trainset, size_t seed, size_t nb_threads) {
    pImpl->train(std::move(trainset), seed, nb_threads);
  }

  classifier::ResultN PF2018::predict(const DTSMap& testset, size_t seed, size_t nb_threads) {
    return pImpl->predict(testset, seed, nb_threads);
  }

} // End of namespace tempo::classifier
