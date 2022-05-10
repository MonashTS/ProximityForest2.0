#include "pf2018.hpp"

#include <tempo/utils/utils.hpp>
#include <tempo/tseries/dataset.hpp>

#include <tempo/classifier/splitting_forest/ipf.hpp>
#include <tempo/classifier/splitting_forest/pftree.hpp>
#include <tempo/classifier/splitting_forest/splitters.hpp>

#include "distance_splitters.hpp"

namespace tempo::classifier {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // PF2018 IMPLEMENTATION
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  struct PF2018::Impl {

    Impl(size_t nb_trees, size_t nb_candidates)
      : nb_trees(nb_trees), nb_candidates(nb_candidates) {}

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Train time structures
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    struct TrainState : pf::IState<TrainState> {
      /// State-full PRNG (prevent copy constructor)
      std::unique_ptr<PRNG> prng;

      /// Specific distance state for the splitters
      pf::DistanceSplitterState distance_splitter_state;

      /// Constructor with a seed for the PRNG
      explicit TrainState(size_t seed) :
        prng(std::make_unique<PRNG>(seed)), distance_splitter_state(false) {}

      /// Constructor with a PRNG
      explicit TrainState(std::unique_ptr<PRNG>&& prng) :
        prng(std::move(prng)), distance_splitter_state(false) {}

      /// Allow move construction
      inline TrainState(TrainState&&) = default;

      /// Interface on_leaf
      inline void on_leaf(const pf::BCMVec& bcmvec) override {
        distance_splitter_state.on_leaf(bcmvec);
      }

      /// Interface branch_fork - transmit the prng (state-full)
      inline TrainState branch_fork(size_t /* bidx */) override {
        return TrainState(std::move(prng));
      }

      /// Interface branch_merge - move prng into this, merge other state into this
      inline void branch_merge(TrainState&& other) override {
        this->prng = std::move(other.prng);
        this->distance_splitter_state.merge(other.distance_splitter_state);
      }

      /// Interface forest_fork - create new prng (fully independent clones)
      inline TrainState forest_fork(size_t /* tree_index */) override {
        size_t new_seed = (*prng)();
        return TrainState(new_seed);
      }

    };

    struct TrainData {

      DTSMap trainset;

      inline explicit TrainData(DTSMap trainset) : trainset(move(trainset)) {}

      /// Concept requirement - train_dataset_map
      inline const DTSMap& train_dataset_map() const { return trainset; }

      /// Concept requirement - get_train_dataset
      inline const DTS get_train_dataset(const std::string& tname) const { return trainset.at(tname); }

      /// Get header
      inline const DatasetHeader& get_header() const { return trainset.begin()->second.header(); }
    };

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Test time structures
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    // Test time state structure - use the same as the train time
    using TestState = TrainState;

    // Test time data structure - "include" all member from TrainData
    struct TestData : public TrainData {

      DTSMap testset;

      // Concept requirement - test_dataset_map
      inline const DTSMap& test_dataset_map() const { return testset; }

      // Concept requirement - get_test_dataset
      inline const DTS get_test_dataset(const std::string& tname) const { return testset.at(tname); }

      TestData(DTSMap trainset, DTSMap testset) : TrainData(std::move(trainset)), testset(std::move(testset)) {}
    };

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Splitter
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    // --- --- --- Splitter parameterization

    /// Exponents array for the cost function - always 2
    inline static const auto exp2 = [](TrainState& /* state */) -> double { return 2; };

    /// Transformation name array - always "default"
    inline static const auto def = [](TrainState& /* state */ ) -> std::string { return "default"; };

    /// Transformation name array - always "d1", the first derivative
    inline static const auto d1 = [](TrainState& /* state */) -> std::string { return "d1"; };

    /// Random window computation function [0, Lmax/4]
    inline static const auto getw = [](TrainState& state, const TrainData& data) -> size_t {
      const size_t win_top = std::ceil((double)data.get_header().length_max()/4.0);
      return std::uniform_int_distribution<size_t>(0, win_top)(*state.prng);
    };

    /// ERP Gap Value *AND* LCSS epsilon. Requires the name of the dataset.
    /// Random fraction of the dataset standard deviation, within [stddev/5, stddev[
    inline static const auto frac_stddev =
      [](TrainState& state, const TrainData& data, const pf::BCMVec& bcmvec, const std::string& tn) -> double {
        const auto& train_dataset = data.get_train_dataset(tn);
        auto stddev_ = stddev(train_dataset, bcmvec.back().to_IndexSet());
        return std::uniform_real_distribution<F>(0.2*stddev_, stddev_)(*state.prng);
      };

    /// List of MSM costs
    inline static const auto msm_cost = [](TrainState& state) -> size_t {
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
      return utils::pick_one(costs, N, *state.prng);
    };

    /// TWE nu parameters
    inline static const auto twe_nu = [](TrainState& state) -> size_t {
      constexpr size_t N = 10;
      double nus[N]{0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1};
      return utils::pick_one(nus, N, *state.prng);
    };

    /// TWE lambda parameters
    inline static const auto twe_lambda = [](TrainState& state) -> size_t {
      constexpr size_t N = 10;
      double lambdas[N]{0, 0.011111111, 0.022222222, 0.033333333, 0.044444444,
                        0.055555556, 0.066666667, 0.077777778, 0.088888889, 0.1};
      return utils::pick_one(lambdas, N, *state.prng);
    };

    // --- --- --- Splitter Builder

    template<typename D>
    using SG1 = pf::SG_1NN<TrainState, TrainData, TestState, TestData, D>;

    /// SQED
    using DA_t = typename pf::DComp_DA::template Generator<TrainState, TrainData>;
    static inline auto sg_1nn_da = std::make_shared<SG1<DA_t>>(DA_t(def, exp2));

    /// DTW Full window
    using DTWFull_t = typename pf::DComp_DTWFull::template Generator<TrainState, TrainData>;
    static inline auto sg_1nn_dtwfull = std::make_shared<SG1<DTWFull_t>>(DTWFull_t(def, exp2));
    static inline auto sg_1nn_ddtwfull = std::make_shared<SG1<DTWFull_t>>(DTWFull_t(d1, exp2));

    /// DTW with parametric window
    using DTW_t = typename pf::DComp_DTW::template Generator<TrainState, TrainData>;
    static inline auto sg_1nn_dtw = std::make_shared<SG1<DTW_t>>(DTW_t(def, exp2, getw));
    static inline auto sg_1nn_ddtw = std::make_shared<SG1<DTW_t>>(DTW_t(d1, exp2, getw));

    /// WDTW with parametric window
    using WDTW_t = typename pf::DComp_WDTW::template Generator<TrainState, TrainData>;
    static inline auto sg_1nn_wdtw = std::make_shared<SG1<WDTW_t>>(WDTW_t(def, exp2));
    static inline auto sg_1nn_wddtw = std::make_shared<SG1<WDTW_t>>(WDTW_t(d1, exp2));

    /// ERP with parametric window and gap value
    using ERP_t = typename pf::DComp_ERP::template Generator<TrainState, TrainData>;
    static inline auto sg_1nn_erp = std::make_shared<SG1<ERP_t>>(ERP_t(def, exp2, getw, frac_stddev));

    /// LCSS with parametric window and epsilon value
    using LCSS_t = typename pf::DComp_LCSS::template Generator<TrainState, TrainData>;
    static inline auto sg_1nn_lcss = std::make_shared<SG1<LCSS_t>>(LCSS_t(def, getw, frac_stddev));

    /// MSM with parametric window and epsilon value
    using MSM_t = typename pf::DComp_MSM::template Generator<TrainState, TrainData>;
    static inline auto sg_1nn_msm = std::make_shared<SG1<MSM_t>>(MSM_t(def, msm_cost));

    /// TWE with parametric window and epsilon value
    using TWE_t = typename pf::DComp_TWE::template Generator<TrainState, TrainData>;
    static inline auto sg_1nn_twe = std::make_shared<SG1<TWE_t>>(TWE_t(def, twe_nu, twe_lambda));

    /// Helper function building a tree trainer.
    /// combines a splitter chooser with a number of candidate and a leaf generator
    static inline std::shared_ptr<pf::PFTreeTrainer<TrainState, TrainData, TestState, TestData>> fun_tree_trainer(
      size_t nbc) {
      auto node = std::make_shared<pf::SG_chooser<TrainState, TrainData, TestState, TestData>>(
        typename pf::SG_chooser<TrainState, TrainData, TestState, TestData>::SGVec_t(
          {
            sg_1nn_da,
            sg_1nn_dtw,
            sg_1nn_ddtw,
            sg_1nn_dtwfull,
            sg_1nn_ddtwfull,
            sg_1nn_wdtw,
            sg_1nn_wddtw,
            sg_1nn_erp,
            sg_1nn_lcss,
            sg_1nn_msm,
            sg_1nn_twe
          }
        ), nbc
      );
      auto leaf = std::make_shared<pf::SGLeaf_PureNode<TrainState, TrainData, TestState, TestData>>();
      // mk trainer
      return std::make_shared<pf::PFTreeTrainer<TrainState, TrainData, TestState, TestData>>(leaf, node);
    }


    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Fields
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    // --- --- --- Valid after construction
    size_t nb_trees;
    size_t nb_candidates;
    // --- --- --- Valid after train
    std::optional<DTSMap> o_trainset;                   // Trainset needed at test time (for distance to exemplar)
    std::vector<TrainState> trained_states;             // Train state per trained tree
    pf::PForest<TestState, TestData> trained_forest;    //


    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Functions
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Train the classifier
    void train(DTSMap trainset, size_t seed, size_t nb_threads) {
      // Arg check
      if (!trainset.contains("default")) { throw std::invalid_argument("'default' transform not found in trainset"); }
      if (!trainset.contains("d1")) { throw std::invalid_argument("'d1' transform not found in trainset"); }
      // Build the state and the forest
      TrainState train_state(seed);
      TrainData train_data(trainset);
      auto tree_trainer = fun_tree_trainer(nb_candidates);
      pf::PForestTrainer<TrainState, TrainData, TestState, TestData> forest_trainer(tree_trainer, nb_trees);
      // Training: requires the ByClassMap of the train
      auto train_dataset = trainset.at("default");
      auto [train_bcm, train_noclass] = train_dataset.header().get_BCM();
      auto [states, forest] = forest_trainer.train(train_state, train_data, train_bcm, nb_threads);
      // Store training result
      o_trainset = {std::move(trainset)};
      trained_states = std::move(states);
      trained_forest = std::move(forest);
    }

    /// Predict probabilities for the testset. Must be called after 'train'!
    void predict(const DTSMap& testset, arma::mat& out_probabilities, arma::rowvec& out_weights,
                 size_t seed, size_t nb_threads) {
      // Check calling logic
      if (!o_trainset) { throw std::logic_error("'predict' called before 'train'"); }
      const auto& trainset = o_trainset.value();
      // Arg check
      if (!testset.contains("default")) { throw std::invalid_argument("'default' transform not found in testset"); }
      if (!testset.contains("d1")) { throw std::invalid_argument("'d1' transform not found in testset"); }
      // Build test state
      TestState test_state(seed);
      TestData test_data(trainset, testset);
      // Configure output
      size_t nb_rows = test_data.get_header().nb_labels();  // i.e. length of a column
      size_t nb_cols = test_data.get_header().size();       // i.e. length of a row
      out_probabilities.zeros(nb_rows, nb_cols);
      out_weights.zeros(nb_cols);
      // For each query
      for (size_t query = 0; query<nb_cols; ++query) {
        // Get cardinalities as double
        arma::Col<double> cards = arma::conv_to<arma::Col<double>>::from(
          trained_forest.predict_cardinality(test_state, test_data, query, nb_threads));
        // Get the total
        double sum = arma::sum(cards);
        // Record result
        arma::colvec proba = cards/sum;
        out_probabilities.col(query) = proba;
        out_weights[query] = sum;
      }
    }

  }; // End of struct PF2018::Impl

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // PF2018 INTERFACE
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  PF2018::PF2018(size_t nb_trees, size_t nb_candidates)
    : pImpl(std::make_unique<Impl>(nb_trees, nb_candidates)) {}

  PF2018::~PF2018() = default;

  void PF2018::train(DTSMap trainset, size_t seed, size_t nb_threads) {
    pImpl->train(std::move(trainset), seed, nb_threads);
  }

  void PF2018::predict(const DTSMap& testset, arma::mat& out_probabilities, arma::rowvec& out_weights,
                       size_t seed, size_t nb_threads) {
    pImpl->predict(testset, out_probabilities, out_weights, seed, nb_threads);
  }

} // End of namespace tempo::classifier
