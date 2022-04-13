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

    /// Exponents array for the cost function - always 2
    inline static const auto exp2 = std::make_shared<std::vector<double>>(std::vector<double>{2});

    /// Transformation name array - always "default"
    inline static const auto def = std::make_shared<std::vector<std::string>>(std::vector<std::string>{"default"});

    /// Transformation name array - always "d1", the first derivative
    inline static const auto d1 = std::make_shared<std::vector<std::string>>(std::vector<std::string>{"d1"});

    /// List of MSM costs
    inline static const auto msm_costs = std::make_shared<std::vector<double>>(
      std::vector<double>{
        0.01, 0.01375, 0.0175, 0.02125, 0.025, 0.02875, 0.0325, 0.03625, 0.04, 0.04375,
        0.0475, 0.05125, 0.055, 0.05875, 0.0625, 0.06625, 0.07, 0.07375, 0.0775, 0.08125,
        0.085, 0.08875, 0.0925, 0.09625, 0.1, 0.136, 0.172, 0.208, 0.244, 0.28, 0.316, 0.352,
        0.388, 0.424, 0.46, 0.496, 0.532, 0.568, 0.604, 0.64, 0.676, 0.712, 0.748, 0.784,
        0.82, 0.856, 0.892, 0.928, 0.964, 1, 1.36, 1.72, 2.08, 2.44, 2.8, 3.16, 3.52, 3.88,
        4.24, 4.6, 4.96, 5.32, 5.68, 6.04, 6.4, 6.76, 7.12, 7.48, 7.84, 8.2, 8.56, 8.92, 9.28,
        9.64, 10, 13.6, 17.2, 20.8, 24.4, 28, 31.6, 35.2, 38.8, 42.4, 46, 49.6, 53.2, 56.8,
        60.4, 64, 67.6, 71.2, 74.8, 78.4, 82, 85.6, 89.2, 92.8, 96.4, 100
      }
    );

    /// TWE nu parameters
    inline static const auto twe_nus = std::make_shared<std::vector<double>>(
      std::vector<double>{0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1}
    );

    /// TWE lambda parameters
    inline static const auto twe_lambdas = std::make_shared<std::vector<double>>(
      std::vector<double>{0, 0.011111111, 0.022222222, 0.033333333, 0.044444444, 0.055555556, 0.066666667, 0.077777778,
                          0.088888889, 0.1}
    );

    /// Structure for the state at both train and test time
    struct PFState :
      public pf::IState<L, PFState>,
      public pf::TimeSeriesDatasetHeader<PFState, F, L> {

      /// Track the depth of the tree
      size_t depth{0};

      /// Track selected distances; On merge, just create a new empty one (done by default)
      DistanceSplitterState<L, true> distance_splitter_state;

      /// Pseudo random number generator: use a unique pointer (stateful)
      std::unique_ptr<PRNG> prng;

      /// Dictionary of name->dataset of time series
      std::shared_ptr<pf::DatasetMap_t<F, L>> dataset_shared_map;

      PFState(size_t seed, std::shared_ptr<pf::DatasetMap_t<F, L>> dataset_shared_map)
        : prng(std::make_unique<PRNG>(seed)), dataset_shared_map(std::move(dataset_shared_map)) {}

      /// Ensure we do not copy the state by error: we have to properly deal with the random number generator
      PFState(const PFState&) = delete;

      /// Ensure that we have a move constructor
      PFState(PFState&&) = default;

    private:

      /// Cloning Constructor: create a new PRNG
      PFState(size_t depth, size_t new_seed, std::shared_ptr<pf::DatasetMap_t<F, L>> map) :
        depth(depth), prng(std::make_unique<PRNG>(new_seed)), dataset_shared_map(std::move(map)) {}

      /// Forking Constructor: transmit PRNG into the new state
      PFState(size_t depth, std::unique_ptr<PRNG>&& m_prng, std::shared_ptr<pf::DatasetMap_t<F, L>> map) :
        depth(depth), prng(std::move(m_prng)), dataset_shared_map(std::move(map)) {}

    public:

      /// Transmit the prng down the branch
      PFState branch_fork(size_t /* bidx */) override {
        return PFState(depth + 1, std::move(prng), dataset_shared_map);
      }

      /// Merge "other" into "this". Move the prng into this. Merge statistics into this
      void branch_merge(PFState&& other) override {
        // Resources: move!
        prng = std::move(other.prng);
        // Values:
        distance_splitter_state.merge(move(other.distance_splitter_state));
        depth = std::max(depth, other.depth);
      }

      /// Clone at the forest level - clones must be fully independent as they can be used in parallel
      /// Create a new prng
      std::unique_ptr<PFState> forest_fork(size_t /* tree_index */) override {
        size_t new_seed = (*prng)();
        return std::unique_ptr<PFState>(new PFState(depth, new_seed, dataset_shared_map));
      }
    };

    /// Test time structure
    class Classifier {

      PFState test_state;
      std::shared_ptr<PForest<L, PFState>> forest;

    public :

      Classifier(
        size_t seed,
        std::shared_ptr<pf::DatasetMap_t<F, L>> dataset_shared_map,
        std::shared_ptr<PForest<L, PFState>> forest
      ) : test_state(seed, dataset_shared_map), forest(std::move(forest)) {}

      [[nodiscard]]
      std::tuple<double, std::vector<double>> predict_proba(size_t index, size_t nbthread) {
        return forest->predict_proba(test_state, index, nbthread);
      }

    };

    /// Result of the train
    struct Trained {

      std::vector<std::unique_ptr<PFState>> trained_states;
      std::shared_ptr<PForest<L, PFState>> trained_forest;

      Trained(
        std::vector<std::unique_ptr<PFState>>&& trained_states,
        std::shared_ptr<PForest<L, PFState>> trained_forest
      ) : trained_states(std::move(trained_states)),
          trained_forest(std::move(trained_forest)) {}

      /// Get a classifier over a test set
      Classifier get_classifier_for(
        size_t seed,
        std::shared_ptr<pf::DatasetMap_t<F, L>> dataset_shared_map
      ) noexcept(false) {

        // Check that we have trained the classifier
        if (!(bool)trained_forest) {
          throw std::logic_error("PF2018 must be trained first");
        }
        // Arg check
        if (!dataset_shared_map->contains("default")) {
          throw std::invalid_argument("'default' transform not found in data_shared_map");
        }
        if (!dataset_shared_map->contains("d1")) {
          throw std::invalid_argument("'d1' transform not found in data_shared_map");
        }

        // Build state and forest trainer
        return Classifier(seed, dataset_shared_map, trained_forest);
      }

    };

  private:

    // --- --- --- --- --- -- --- --- -- --- --- -- --- --- -- --- --- -- --- --- -- --- --- -- --- --- -- --- --- --
    // List of splitters. This one are shared amongst all PF instances.
    // --- --- --- --- --- -- --- --- -- --- --- -- --- --- -- --- --- -- --- --- -- --- --- -- --- --- -- --- --- --

    /// SQED
    static inline std::shared_ptr<pf::SG_1NN_DA<F, L, PFState, PFState>> sg_1nn_da =
      std::make_shared<pf::SG_1NN_DA<F, L, PFState, PFState>>(def, exp2);

    /// DTW Full Window
    static inline std::shared_ptr<pf::SG_1NN_DTWFull<F, L, PFState, PFState>> sg_1nn_dtwf =
      std::make_shared<pf::SG_1NN_DTWFull<F, L, PFState, PFState>>(def, exp2);

    /// DDTW Full Window
    static inline std::shared_ptr<pf::SG_1NN_DTWFull<F, L, PFState, PFState>> sg_1nn_ddtwf =
      std::make_shared<pf::SG_1NN_DTWFull<F, L, PFState, PFState>>(d1, exp2);

    /// DTW Window
    static inline std::shared_ptr<pf::SG_1NN_DTW<F, L, PFState, PFState>> sg_1nn_dtw =
      std::make_shared<pf::SG_1NN_DTW<F, L, PFState, PFState>>(def, exp2);

    /// DDTW Window
    static inline std::shared_ptr<pf::SG_1NN_DTW<F, L, PFState, PFState>> sg_1nn_ddtw =
      std::make_shared<pf::SG_1NN_DTW<F, L, PFState, PFState>>(d1, exp2);

    /// WDTW
    static inline std::shared_ptr<pf::SG_1NN_WDTW<F, L, PFState, PFState>> sg_1nn_wdtw =
      std::make_shared<pf::SG_1NN_WDTW<F, L, PFState, PFState>>(def, exp2);

    /// WDDTW
    static inline std::shared_ptr<pf::SG_1NN_WDTW<F, L, PFState, PFState>> sg_1nn_wddtw =
      std::make_shared<pf::SG_1NN_WDTW<F, L, PFState, PFState>>(d1, exp2);

    /// ERP
    static inline std::shared_ptr<pf::SG_1NN_ERP<F, L, PFState, PFState>> sg_1nn_erp =
      std::make_shared<pf::SG_1NN_ERP<F, L, PFState, PFState>>(def, exp2);

    /// LCSS
    static inline std::shared_ptr<pf::SG_1NN_LCSS<F, L, PFState, PFState>> sg_1nn_lcss =
      std::make_shared<pf::SG_1NN_LCSS<F, L, PFState, PFState>>(def);

    /// MSM
    static inline std::shared_ptr<pf::SG_1NN_MSM<F, L, PFState, PFState>> sg_1nn_msm =
      std::make_shared<pf::SG_1NN_MSM<F, L, PFState, PFState>>(def, msm_costs);

    /// TWE
    static inline std::shared_ptr<pf::SG_1NN_TWE<F, L, PFState, PFState>> sg_1nn_twe =
      std::make_shared<pf::SG_1NN_TWE<F, L, PFState, PFState>>(def, twe_nus, twe_lambdas);

    /// Leaf generator
    static inline std::shared_ptr<pf::SGLeaf_PureNode<F, L, PFState, PFState>> sgleaf_purenode =
      std::make_shared<pf::SGLeaf_PureNode<F, L, PFState, PFState>>();

    /// Helper function building a tree trainer, combining a splitter chooser with a number of candidate and
    /// our leaf generator
    static std::shared_ptr<pf::PFTreeTrainer<L, PFState, PFState>> tree_trainer(size_t nbc) {
      auto chooser = std::make_shared<pf::SG_chooser<L, PFState, PFState>>(
        typename pf::SG_chooser<L, PFState, PFState>::SGVec_t(
          {sg_1nn_da, sg_1nn_dtwf, sg_1nn_ddtwf, sg_1nn_dtw, sg_1nn_ddtw, sg_1nn_wdtw, sg_1nn_wddtw,
           sg_1nn_erp, sg_1nn_lcss, sg_1nn_msm, sg_1nn_twe}
        ), nbc
      );
      return std::make_shared<pf::PFTreeTrainer<L, PFState, PFState>>(sgleaf_purenode, std::move(chooser));
    }

    size_t _nbtree;
    size_t _nbcandidate;
    std::shared_ptr<pf::PFTreeTrainer<L, PFState, PFState>> _tree_trainer;

  public:

    /// Build a new PF2018 classifier trainer
    PF2018(size_t nbtree, size_t nbcandidate) :
      _nbtree(nbtree),
      _nbcandidate(nbcandidate),
      _tree_trainer(tree_trainer(nbcandidate)) {}

    /// Train the classifier
    Trained train(
      size_t seed,
      std::shared_ptr<pf::DatasetMap_t<F, L>> dataset_shared_map,
      size_t nbthreads
    ) noexcept(false) {
      const auto total_train_start = utils::now();

      // Arg check
      if (!dataset_shared_map->contains("default")) {
        throw std::invalid_argument("'default' transform not found in data_shared_map");
      }
      if (!dataset_shared_map->contains("d1")) {
        throw std::invalid_argument("'d1' transform not found in data_shared_map");
      }

      // Build state and forest trainer
      PFState train_state(seed, dataset_shared_map);
      pf::PForestTrainer<L, PFState, PFState> forest_trainer(_tree_trainer, _nbtree);

      // Build ByClassMap vector
      auto train_dataset = dataset_shared_map->at("default");
      auto[train_bcm, train_noclass] = train_dataset.header().get_BCM();
      std::vector<ByClassMap<L>> train_bcmvec{train_bcm};

      // training...
      auto[trained_states, trained_forest] = forest_trainer.train(train_state, train_bcmvec, nbthreads);

      const auto total_train_delta = utils::now() - total_train_start;
      std::cout << "Total train time = " << utils::as_string(total_train_delta) << std::endl;

      return Trained(std::move(trained_states), std::move(trained_forest));
    }

  }; // End of struct PF2018

} // End of namespace libtempo::classifier::pf



