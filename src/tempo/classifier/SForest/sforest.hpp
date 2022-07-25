#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/dts.hpp>
#include <tempo/classifier/utils.hpp>

#include "stree.hpp"

namespace tempo::classifier::SForest {

  // --- --- --- --- --- ---
  // State Concepts

  /// ForestState concept: a state with a couple of fork/merge methods at the forest level
  template<typename S>
  concept ForestState = requires(S& s0){

    requires requires(size_t tree_idx){
      { s0.forest_fork(tree_idx) } -> std::same_as<S>;
    };

    requires requires(S&& s1){
      s0.forest_merge(std::move(s1));
    };
  };

  // --- --- --- --- --- ---
  // Forest implementation

  /// Test time splitting forest
  template<ForestState TestS, TestData TestD>
  struct SForest {

    // Shorthand
    using TreeVec = std::vector<std::unique_ptr<SNode<TestS, TestD>>>;

    /// Vector of trees forming the forest
    TreeVec forest;

    /// Number of train class for which this forest has been trained
    size_t trainclass_cardinality{};

    // --- --- ---

    SForest() = default;

    SForest(SForest const& other) = delete;
    SForest& operator =(SForest const& other) = delete;

    SForest(SForest&& other) noexcept = default;
    SForest& operator =(SForest&& other) noexcept = default;

    SForest(TreeVec&& forest, size_t trainclass_cardinality) :
      forest(std::move(forest)), trainclass_cardinality(trainclass_cardinality) {}

    // --- --- ----

    /// Classification result. Transmit the state back and produces a classifier::Result
    struct R {
      std::unique_ptr<TestS> state;
      classifier::Result1 result;
    };

    /// Given a testing state and testing data, do a prediction for the exemplar 'index'
    R predict(std::unique_ptr<TestS> state, TestD const& data, size_t test_index, size_t nb_threads) const {
      using NodeRes = typename SNode<TestS, TestD>::R;
      const size_t nbtree = forest.size();

      // Result variable
      classifier::Result1 result(trainclass_cardinality);

      // Multithreading control
      std::mutex mutex;

      auto test_task = [&](size_t tree_index) {
        // Generate local state - mutex!
        std::unique_ptr<TestS> local_state;
        {
          std::lock_guard lock(mutex);
          local_state = std::make_unique<TestS>(state->forest_fork(tree_index));
        }
        // Testing...
        NodeRes r = forest[tree_index]->predict(std::move(local_state), data, test_index);
        // Accumulate weighted probabilities and merge state back
        {
          std::lock_guard lock(mutex);
          state->forest_merge(std::move(*r.state));
          result.probabilities += r.result.probabilities*r.result.weight;
          result.weight += r.result.weight;
        }
      };

      // Create the tasks per tree. Note that we clone the state.
      tempo::utils::ParTasks p;
      for (size_t i = 0; i<nbtree; ++i) { p.push_task(test_task, i); }
      p.execute(nb_threads);

      result.probabilities /= result.weight;

      // Build & return result
      return R{std::move(state), std::move(result)};
    }
  };

  template<ForestState TrainS, TrainData TrainD, typename TestS, typename TestD>
  struct SForestTrainer {
    using TreeTrainer = STreeTrainer<TrainS, TrainD, TestS, TestD>;

    std::shared_ptr<const TreeTrainer> tree_trainer;
    size_t nbtrees;

    SForestTrainer(std::shared_ptr<TreeTrainer> tree_trainer, size_t nbtrees) :
      tree_trainer(std::move(tree_trainer)),
      nbtrees(nbtrees) {}


    // --- --- ----

    /// Training result. Transmit the state back and produces a forest
    struct R {
      std::unique_ptr<TestS> state;
      std::unique_ptr<SForest<TestS, TestD>> forest;
    };

    /// Training a forest by training each tree individually
    R train(std::unique_ptr<TrainS> state, const TrainD& data, ByClassMap const& bcm, size_t nb_threads) {

      typename SForest<TestS, TestD>::TreeVec forest;
      forest.reserve(nbtrees);

      // Multithreading control
      std::mutex mutex;

      auto mk_task = [&](size_t tree_index) {

        std::unique_ptr<TrainS> local_state;
        {
          std::lock_guard lock(mutex);
          std::cout << "Start tree " << tree_index << std::endl;
          local_state = std::make_unique<TrainS>(state->forest_fork(tree_index));
        }

        auto start = tempo::utils::now();
        typename TreeTrainer::R r = tree_trainer->train(std::move(local_state), data, bcm);
        auto delta = tempo::utils::now() - start;
        {
          // Lock protecting the forest and out printing
          std::lock_guard lock(mutex);
          // --- Merge state back
          state->forest_merge(std::move(*r.state));
          // --- Printing
          auto cf = std::cout.fill();
          std::cout << std::setfill('0');
          std::cout << std::setw(3) << tree_index + 1 << " / " << nbtrees << "   ";
          std::cout << std::setw(3) << "Depth = " << r.tree->depth() << "   ";
          std::cout << std::setw(3) << "Nb nodes = " << r.tree->nb_nodes() << "   ";
          std::cout.fill(cf);
          std::cout << " timing: " << tempo::utils::as_string(delta) << std::endl;
          // --- Add in the forest
          forest.push_back(std::move(r.tree));
        }
      };

      // Create the tasks per tree. Note that we clone the state.
      tempo::utils::ParTasks p;
      for (size_t i = 0; i<nbtrees; ++i) { p.push_task(mk_task, i); }
      p.execute(nb_threads);

      // Build result & return
      return {
        std::move(state),
        std::make_unique<SForest<TestS, TestD>>(std::move(forest), data.get_train_header().nb_classes())
      };

    }
  };

} // End of namespace tempo::classifier::SForest
