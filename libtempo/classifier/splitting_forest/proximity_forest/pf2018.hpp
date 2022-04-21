#pragma once

#include "libtempo/classifier/splitting_forest/pftree.hpp"
#include "distance_splitters.hpp"
#include "libtempo/classifier/splitting_forest/splitters.hpp"

namespace libtempo::classifier::pf {

  /** Implementation of Proximity Forest 2018 in the tempo framework */
  template<Label L>
  struct PF2018 {

    /// Floating Point type
    using F = double;

    /// Pseudo Random Number Generator
    using PRNG = std::mt19937_64;

    /// Train time state structure
    struct TrainState : public pf::IState<L, TrainState> {

      /// Track the depth of the tree; starts at 1
      size_t current_depth{1};
      size_t max_depth{0};

      /// Track selected distances; On merge, just create a new empty one (done by default)
      DistanceSplitterState<L> distance_splitter_state;

      /// Pseudo random number generator: use a unique pointer (stateful)
      std::unique_ptr<PRNG> prng;

      TrainState(bool do_instrumentation, size_t seed) :
        distance_splitter_state(do_instrumentation),
        prng(std::make_unique<PRNG>(seed)) {}

      /// Ensure we do not copy the state by error: we have to properly deal with the random number generator
      TrainState(const TrainState&) = delete;

      /// Ensure that we have a move constructor
      TrainState(TrainState&&) = default;

    private:

      TrainState(bool do_instrumentation,
                 size_t current_depth,
                 size_t seed) :
        current_depth(current_depth),
        distance_splitter_state(do_instrumentation),
        prng(std::make_unique<PRNG>(seed)) {}

      /// Forking Constructor: transmit PRNG into the new state
      TrainState(bool do_instrumentation,
                 size_t current_depth,
                 std::unique_ptr<PRNG>&& m_prng) :
        current_depth(current_depth),
        distance_splitter_state(do_instrumentation),
        prng(std::move(m_prng)) {}

    public:

      /// On leaf
      void on_leaf(const BCMVec<L>& bcmvec) override {
        max_depth = current_depth;
        distance_splitter_state.on_leaf(bcmvec);
      }

      /// Transmit the prng down the branch
      TrainState branch_fork(size_t /* bidx */) override {
        return TrainState(distance_splitter_state.do_instrumentation, current_depth + 1, std::move(prng));
      }

      /// Merge "other" into "this". Move the prng into this. Merge statistics into this
      void branch_merge(TrainState&& other) override {
        // Resources: move!
        prng = std::move(other.prng);
        // Values:
        distance_splitter_state.merge(move(other.distance_splitter_state));
        max_depth = std::max(max_depth, other.max_depth);
      }

      /// Clone at the forest level - clones must be fully independent as they can be used in parallel
      /// Create a new prng
      std::unique_ptr<TrainState> forest_fork(size_t /* tree_index */) override {
        size_t new_seed = (*prng)();
        return std::unique_ptr<TrainState>(
          new TrainState(distance_splitter_state.do_instrumentation, current_depth, new_seed)
        );
      }
    };

    /// Train time data structure
    struct TrainData {

      /// Dictionary of name->dataset of time series
      std::shared_ptr<pf::DatasetMap_t<F, L>> train_dataset_map_sptr;

      /// Concept requirement - train_dataset_map
      [[nodiscard]] inline
      const pf::DatasetMap_t<F, L>& train_dataset_map() const { return *train_dataset_map_sptr; }

      /// Concept requirement - get_train_dataset
      [[nodiscard]] inline
      const DTS<F, L> get_train_dataset(const std::string& tname) const {
        return train_dataset_map().at(tname);
      }

      /// Get header
      const DatasetHeader<L>& get_header() const {
        return train_dataset_map().begin()->second.header();
      }

      explicit TrainData(std::shared_ptr<pf::DatasetMap_t<F, L>> dataset_shared_map) :
        train_dataset_map_sptr(dataset_shared_map) {}
    };

    /// Test time state structure - use the same as the train time
    using TestState = TrainState;

    /// Test time data structure - "include" all member from TrainData
    struct TestData : public TrainData {

      /// Dictionary of name->dataset of time series
      std::shared_ptr<pf::DatasetMap_t<F, L>> test_dataset_map_sptr;

      /// Concept requirement - test_dataset_map
      [[nodiscard]] inline
      const pf::DatasetMap_t<F, L>& test_dataset_map() const { return *test_dataset_map_sptr; }

      /// Concept requirement - get_test_dataset
      [[nodiscard]] inline
      const DTS<F, L> get_test_dataset(const std::string& tname) const {
        return test_dataset_map().at(tname);
      }

      TestData(
        std::shared_ptr<pf::DatasetMap_t<F, L>> train_dataset_shared_map,
        std::shared_ptr<pf::DatasetMap_t<F, L>> test_dataset_shared_map
      ) :
        TrainData(std::move(train_dataset_shared_map)),
        test_dataset_map_sptr(test_dataset_shared_map) {}
    };

    /// Test time structure
    class Classifier {

      TestState test_state;
      TestData test_data;
      std::shared_ptr<PForest<L, TestState, TestData>> forest;

    public :

      /// Build a classifier with a seed, the train data, the test data and the forest
      Classifier(
        size_t seed,
        std::shared_ptr<pf::DatasetMap_t<F, L>> train_dataset_shared_map,
        std::shared_ptr<pf::DatasetMap_t<F, L>> test_dataset_shared_map,
        std::shared_ptr<PForest<L, TestState, TestData>> forest,
        bool do_instrumentation
      ) :
        test_state(do_instrumentation, seed),
        test_data(train_dataset_shared_map, test_dataset_shared_map),
        forest(std::move(forest)) {}

      /// Classifier interface
      [[nodiscard]]
      std::tuple<double, std::vector<double>>
      predict_proba(size_t index, size_t nbthread) {
        return forest->predict_proba(test_state, test_data, index, nbthread);
      }

    };

    /// Result of the train
    struct Trained {

      std::shared_ptr<pf::DatasetMap_t<F, L>> train_dataset_shared_map;
      std::vector<std::unique_ptr<TrainState>> trained_states;
      std::shared_ptr<PForest<L, TestState, TestData>> trained_forest;

      Trained(
        std::shared_ptr<pf::DatasetMap_t<F, L>> train_dataset_shared_map,
        std::vector<std::unique_ptr<TrainState>>&& trained_states,
        std::shared_ptr<PForest<L, TestState, TestData>> trained_forest
      ) :
        train_dataset_shared_map(std::move(train_dataset_shared_map)),
        trained_states(std::move(trained_states)),
        trained_forest(std::move(trained_forest)) {}

      /// Get a classifier over a test set
      Classifier get_classifier_for(
        size_t seed,
        std::shared_ptr<pf::DatasetMap_t<F, L>> test_dataset_shared_map,
        bool do_instrumentation
      ) noexcept(false) {

        // Check that we have trained the classifier
        if (!(bool)trained_forest) {
          throw std::logic_error("PF2018 must be trained first");
        }
        // Arg check
        if (!test_dataset_shared_map->contains("default")) {
          throw std::invalid_argument("'default' transform not found in data_shared_map");
        }
        if (!test_dataset_shared_map->contains("d1")) {
          throw std::invalid_argument("'d1' transform not found in data_shared_map");
        }

        // Build state and forest trainer
        return Classifier(seed, train_dataset_shared_map, test_dataset_shared_map, trained_forest, do_instrumentation);
      }

    };

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // PROXIMITY FOREST PARAMETERIZATION
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Exponents array for the cost function - always 2
    inline static const auto exp2 = [](TrainState& /* state */) { return 2; };

    /// Transformation name array - always "default"
    inline static const auto def = [](TrainState& /* state */ ) { return "default"; };

    /// Transformation name array - always "d1", the first derivative
    inline static const auto d1 = [](TrainState& /* state */) { return "d1"; };

    /// Random window computation function
    inline static const auto getw = [](TrainState& state, const TrainData& data) -> size_t {
      const size_t win_top = (data.get_header().length_max() + 1)/4;
      return std::uniform_int_distribution<size_t>(0, win_top)(*state.prng);
    };

    /// List of MSM costs
    inline static const auto msm_cost = [](TrainState& state) {
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
    inline static const auto twe_nu = [](TrainState& state) {
      constexpr size_t N = 10;
      double nus[N]{0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1};
      return utils::pick_one(nus, N, *state.prng);
    };

    /// TWE lambda parameters
    inline static const auto twe_lambda = [](TrainState& state) {
      constexpr size_t N = 10;
      double lambdas[N]{0, 0.011111111, 0.022222222, 0.033333333, 0.044444444,
                        0.055555556, 0.066666667, 0.077777778, 0.088888889, 0.1};
      return utils::pick_one(lambdas, N, *state.prng);
    };

  private:

    // --- --- --- --- --- -- --- --- -- --- --- -- --- --- -- --- --- -- --- --- -- --- --- -- --- --- -- --- --- --
    // List of splitters. This one are shared amongst all PF instances.
    // --- --- --- --- --- -- --- --- -- --- --- -- --- --- -- --- --- -- --- --- -- --- --- -- --- --- -- --- --- --
    template<typename D>
    using SG1 = pf::SG_1NN<F, L, TrainState, TrainData, TestState, TestData, D>;

    /// SQED
    using DA_t = typename Splitter_1NN_DA<F, L>::template Generator<TrainState, TrainData>;
    static inline auto sg_1nn_da = std::make_shared<SG1<DA_t>>(DA_t(def, exp2));

    /// DTW Full window
    using DTWFull_t = typename Splitter_1NN_DTWFull<F, L>::template Generator<TrainState, TrainData>;
    static inline auto sg_1nn_dtwfull = std::make_shared<SG1<DTWFull_t>>(DTWFull_t(def, exp2));
    static inline auto sg_1nn_ddtwfull = std::make_shared<SG1<DTWFull_t>>(DTWFull_t(d1, exp2));

    /// DTW with parametric window
    using DTW_t = typename Splitter_1NN_DTW<F, L>::template Generator<TrainState, TrainData>;
    static inline auto sg_1nn_dtw = std::make_shared<SG1<DTW_t>>(DTW_t(def, exp2, getw));
    static inline auto sg_1nn_ddtw = std::make_shared<SG1<DTW_t>>(DTW_t(d1, exp2, getw));

    /// WDTW with parametric window
    using WDTW_t = typename Splitter_1NN_WDTW<F, L>::template Generator<TrainState, TrainData>;
    static inline auto sg_1nn_wdtw = std::make_shared<SG1<WDTW_t>>(WDTW_t(def, exp2));
    static inline auto sg_1nn_wddtw = std::make_shared<SG1<WDTW_t>>(WDTW_t(d1, exp2));

    /// ERP with parametric window and gap value
    using ERP_t = typename Splitter_1NN_ERP<F, L>::template Generator<TrainState, TrainData>;
    static inline auto sg_1nn_erp = std::make_shared<SG1<ERP_t>>(ERP_t(def, exp2, getw));

    /// LCSS with parametric window and epsilon value
    using LCSS_t = typename Splitter_1NN_LCSS<F, L>::template Generator<TrainState, TrainData>;
    static inline auto sg_1nn_lcss = std::make_shared<SG1<LCSS_t>>(LCSS_t(def, getw));

    /// MSM with parametric window and epsilon value
    using MSM_t = typename Splitter_1NN_MSM<F, L>::template Generator<TrainState, TrainData>;
    static inline auto sg_1nn_msm = std::make_shared<SG1<MSM_t>>(MSM_t(def, msm_cost));

    /// TWE with parametric window and epsilon value
    using TWE_t = typename Splitter_1NN_TWE<F, L>::template Generator<TrainState, TrainData>;
    static inline auto sg_1nn_twe = std::make_shared<SG1<TWE_t>>(TWE_t(def, twe_nu, twe_lambda));

    /// Leaf generator
    static inline std::shared_ptr<pf::SGLeaf_PureNode<L, TrainState, TrainData, TestState, TestData>> sgleaf_purenode =
      std::make_shared<pf::SGLeaf_PureNode<L, TrainState, TrainData, TestState, TestData>>();

    /// Helper function building a tree trainer, combining a splitter chooser with a number of candidate and
    /// our leaf generator
    static std::shared_ptr<pf::PFTreeTrainer<L, TrainState, TrainData, TestState, TestData>> tree_trainer(size_t nbc) {
      auto chooser = std::make_shared<pf::SG_chooser<L, TrainState, TrainData, TestState, TestData>>(
        typename pf::SG_chooser<L, TrainState, TrainData, TestState, TestData>::SGVec_t(
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
      // mk trainer
      return std::make_shared<pf::PFTreeTrainer<L, TrainState, TrainData, TestState, TestData>>(sgleaf_purenode
                                                                                                , std::move(chooser));
    }

    size_t _nbtree;
    size_t _nbcandidate;
    std::shared_ptr<pf::PFTreeTrainer<L, TrainState, TrainData, TestState, TestData>> _tree_trainer;

  public:

    /// Build a new PF2018 classifier trainer
    PF2018(size_t nbtree, size_t nbcandidate) :
      _nbtree(nbtree),
      _nbcandidate(nbcandidate),
      _tree_trainer(tree_trainer(nbcandidate)) {}

    /// Train the classifier
    Trained train(
      size_t seed,
      std::shared_ptr<pf::DatasetMap_t<F, L>> train_dataset_shared_map,
      size_t nbthreads = 1,
      bool do_instrumentation = true
    ) noexcept(false) {
      const auto total_train_start = utils::now();

      // Arg check
      if (!train_dataset_shared_map->contains("default")) {
        throw std::invalid_argument("'default' transform not found in data_shared_map");
      }
      if (!train_dataset_shared_map->contains("d1")) {
        throw std::invalid_argument("'d1' transform not found in data_shared_map");
      }

      // Build state and forest trainer
      TrainState train_state(do_instrumentation, seed);
      TrainData train_data(train_dataset_shared_map);
      pf::PForestTrainer<L, TrainState, TrainData, TestState, TestData> forest_trainer(_tree_trainer, _nbtree);

      // Build ByClassMap vector
      auto train_dataset = train_dataset_shared_map->at("default");
      auto [train_bcm, train_noclass] = train_dataset.header().get_BCM();
      std::vector<ByClassMap<L>> train_bcmvec{train_bcm};

      // training...
      auto [trained_states, trained_forest] = forest_trainer.train(train_state, train_data, train_bcmvec, nbthreads);

      const auto total_train_delta = utils::now() - total_train_start;
      std::cout << "Total train time = " << utils::as_string(total_train_delta) << std::endl;

      return Trained(std::move(train_dataset_shared_map), std::move(trained_states), std::move(trained_forest));
    }

  }; // End of struct PF2018

} // End of namespace libtempo::classifier::pf



