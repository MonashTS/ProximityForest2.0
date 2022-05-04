#pragma once

#include <libtempo/utils/utils.hpp>
#include <libtempo/classifier/splitting_forest/ipf.hpp>

#include <functional>
#include <iostream>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace libtempo::classifier::pf {

  /** Proximity Tree at test time
   *  Use an instance of PFTreeTrainer to obtain a trained PFTree, which can then be used to classify instances.
   *  Note: This class implements a node. A tree is represented by the root node.
   * @tparam Stest     Test State type. Must contain the info required by the splitters.
   */
  template<typename TestState, typename TestData>
  struct PFTree {
    using LeafSplitter = std::unique_ptr<IPF_LeafSplitter<TestState, TestData>>;
    using InnerNodeSplitter = std::unique_ptr<IPF_NodeSplitter<TestState, TestData>>;
    using Branches = std::vector<std::unique_ptr<PFTree<TestState, TestData>>>;

    struct InnerNode {
      InnerNodeSplitter splitter;
      Branches branches;
      std::vector<arma::Col<size_t>> cardinalities;
    };

    /** Leaf Node xor Inner Node */
    std::variant<LeafSplitter, InnerNode> node;

  private:

    static void predict_cardinality(const PFTree& pt,
                                    TestState& state,
                                    const TestData& data,
                                    size_t query_idx,
                                    std::vector<arma::Col<size_t>>& out) {
      switch (pt.node.index()) {
      case 0: {
        const LeafSplitter& node = std::get<0>(pt.node);
        out.push_back(node->predict_cardinality(state, data, query_idx));
        break;
      }
      case 1: {
        const InnerNode& n = std::get<1>(pt.node);
        size_t branch_idx = n.splitter->get_branch_index(state, data, query_idx);
        out.push_back(n.cardinalities[branch_idx]);
        const auto& sub = n.branches.at(branch_idx);    // Use at because it is 'const', when [ ] is not
        predict_cardinality(*sub, state, data, query_idx, out);
        break;
      }
      default: utils::should_not_happen();
      }
    }

    /** Main stateless classification function.
     * Given a PFTree node 'pt', recursively call the splitters/follow the branches until reaching a leaf node.
     * @param pt            A node of the tree (initial call should be done on the root node)
     * @param state         Test time state
     * @param data          Test time data
     * @param query_idx     Index of the exemplar in the test dataset
     * @return A vector of probabilities. See DatasetHeader.
     */
    [[nodiscard]]
    static std::tuple<double, std::vector<double>>
    predict_proba(const PFTree& pt, TestState& state, const TestData& data, size_t query_idx) {
      switch (pt.node.index()) {
      case 0: { return std::get<0>(pt.node)->predict_proba(state, data, query_idx); }
      case 1: {
        const InnerNode& n = std::get<1>(pt.node);
        size_t branch_idx = n.splitter->get_branch_index(state, data, query_idx);
        const auto& sub = n.branches.at(branch_idx);    // Use at because it is 'const', when [ ] is not
        return predict_proba(*sub, state, data, query_idx);
      }
      default: utils::should_not_happen();
      }
    }

  public:

    /** Classification function
     * @param state     Test time state. Give access to a mutable state
     * @param data      Test time data. Give access to read only data
     * @param index     Index of the query in the test dataset
     * @return  Vector of probability. See DatasetHeader.
     */
    [[nodiscard]]
    std::tuple<double, std::vector<double>> predict_proba(TestState& state, const TestData& data, size_t index) const {
      return predict_proba(*this, state, data, index);
    }

    [[nodiscard]]
    std::vector<arma::Col<size_t>>
    predict_cardinality(TestState& state, const TestData& data, size_t query_index) const {
      std::vector<arma::Col<size_t>> out;
      predict_cardinality(*this, state, data, query_index, out);
      return out;
    }

  }; // End of PTree


  /** Train time Proximity tree */
  template<typename TrainState, typename TrainData, typename TestState, typename TestData>
  struct PFTreeTrainer {

    // Shorthand for result type
    using Leaf = ResLeaf<TrainState, TrainData, TestState, TestData>;
    using Node = ResNode<TrainState, TrainData, TestState, TestData>;
    using Result = std::variant<Leaf, Node>;

    // Shorthands for Lead and Node generator types
    using LGen = IPF_LeafGenerator<TrainState, TrainData, TestState, TestData>;
    using NGen = IPF_NodeGenerator<TrainState, TrainData, TestState, TestData>;

    // Shorthand for the final tree type
    using PFTree_t = PFTree<TestState, TestData>;

    std::shared_ptr<LGen> leaf_generator;
    std::shared_ptr<NGen> node_generator;

    /// Build a proximity tree trainer with a leaf generator and a node generator
    PFTreeTrainer(std::shared_ptr<LGen> leaf_generator, std::shared_ptr<NGen> node_generator) :
      leaf_generator(leaf_generator), node_generator(node_generator) {}

  private:

    /** Generate a new splitter from a training state and the ByClassMap at the node.
    * @param state  Training state - mutable reference!
    * @param data   Training data - read only reference
    * @param bcmvec stack of BCM from root to this node: 'bcmvec.back()' stands for the BCM at this node.
    * @return  ISplitterGenerator::Result with the splitter and the associated split of the train data
    */
    Result generate(TrainState& state, const TrainData& data, const BCMVec& bcmvec) const {
      std::optional<Leaf> oleaf = leaf_generator->generate(state, data, bcmvec);
      if (oleaf.has_value()) {
        return Result{std::move(oleaf.value())};
      } else {
        return Result{node_generator->generate(state, data, bcmvec)};
      }
    }

  public:

    /// Train a tree
    [[nodiscard]] std::unique_ptr<PFTree_t> train(TrainState& state, const TrainData& data, BCMVec bcmvec) const {
      // Ensure that we have at least one class reaching this node!
      // Note: there may be no data point associated to the class.
      const auto bcm = bcmvec.back();
      assert(bcm.nb_classes()>0);
      // Final return 'ret' variable
      std::unique_ptr<PFTree_t> ret;

      // Call the generator and analyse the result
      Result result = generate(state, data, bcmvec);

      switch (result.index()) {
      case 0: { // Leaf case - stop the recursion, build a leaf node
        Leaf leaf = std::get<0>(std::move(result));
        // Result callback
        leaf.callback(state, data, bcmvec);
        // State callback
        state.on_leaf(bcmvec);
        // Build the leaf node with the splitter
        ret = std::unique_ptr<PFTree_t>(new PFTree_t{.node=std::move(leaf.splitter)});
        break;
      }
      case 1: { // Inner node case: recursion per branch
        Node inner_node = std::get<1>(std::move(result));
        // Result callback
        inner_node.callback(state, data, bcmvec);
        // Build subbranches, then we'll build the current node
        const size_t nbbranches = inner_node.branch_splits.size();
        typename PFTree_t::Branches subbranches;
        subbranches.reserve(nbbranches);
        // Also build the cardinalities -- requires access to the header
        std::vector<arma::Col<size_t>> cardinalities;
        cardinalities.reserve(nbbranches);
        const auto& header = data.get_header();
        // Building loop
        for (size_t idx = 0; idx<nbbranches; ++idx) {
          // Clone state, push bcm
          bcmvec.template emplace_back(std::move(inner_node.branch_splits[idx]));
          // Cardinalities
          const auto& bcm = bcmvec.back();
          arma::Col<size_t> card = get_class_cardinalities(header, bcm);
          cardinalities.push_back(std::move(card));
          // For skate and train sub tree
          TrainState sub_state = state.branch_fork(idx);
          subbranches.push_back(train(sub_state, data, bcmvec));
          // Merge state, pop bcm
          state.branch_merge(std::move(sub_state));
          bcmvec.pop_back();
        }
        ret = std::unique_ptr<PFTree_t>(
          new PFTree_t{
            .node = typename PFTree_t::InnerNode{
              .splitter = std::move(inner_node.splitter),
              .branches = std::move(subbranches),
              .cardinalities = std::move(cardinalities)
            }
          }
        );

        break;
      }

      default: utils::should_not_happen();
      }

      return ret;
    }
  };
  // End of struct PFTreeTrainer


  /** Proximity Forest at test time */
  template<typename TestState, typename TestData>
  struct PForest {
    using TreeVec = std::vector<std::unique_ptr<PFTree<TestState, TestData>>>;

    TreeVec forest;
    arma::Col<size_t> train_class_cardinalities;

    PForest(TreeVec&& forest, arma::Col<size_t> cardinaliaties) :
      forest(std::move(forest)), train_class_cardinalities(std::move(cardinaliaties)) {}

    arma::Col<size_t> predict_cardinality(TestState& state, const TestData& data, size_t instance_index,
                                          size_t nbthread) const {
      const size_t nbtree = forest.size();

      // Result variables
      arma::Col<size_t> result(train_class_cardinalities.size());

      // State vector
      std::vector<std::unique_ptr<TestState>> states_vec;
      states_vec.reserve(nbtree);

      // Multithreading control
      std::mutex mutex;

      auto test_task = [&](size_t tree_index) {
        std::vector<arma::Col<size_t>> out = forest[tree_index]->predict_cardinality(state, data, instance_index);
        { // Accumulate weighted probabilities
          std::lock_guard lock(mutex);
          result += out.back();
        }
      };

      // Create the tasks per tree. Note that we clone the state.
      libtempo::utils::ParTasks p;
      for (size_t i = 0; i<nbtree; ++i) {
        states_vec.push_back(state.forest_fork(i));
        p.push_task(test_task, i);
      }

      p.execute(nbthread);

      return result;
    }
  };

  /** Proximity Forest at train time */
  template<typename TrainState, typename TrainData, typename TestState, typename TestData>
  struct PForestTrainer {

    std::shared_ptr<const PFTreeTrainer<TrainState, TrainData, TestState, TestData>> tree_trainer;
    size_t nbtrees;

    PForestTrainer(
      std::shared_ptr<PFTreeTrainer<TrainState, TrainData, TestState, TestData>> tree_trainer,
      size_t nbtrees
    ) :
      tree_trainer(std::move(tree_trainer)),
      nbtrees(nbtrees) {}

    /** Train the proximity forest */
    std::tuple<
      std::vector<std::unique_ptr<TrainState>>,
      std::shared_ptr<PForest<TestState, TestData>>
    > train(TrainState& state, const TrainData& data, ByClassMap bcm, size_t nbthread) {

      std::mutex mutex;

      BCMVec bcmvec{bcm};
      const auto& header = data.get_header();
      arma::Col<size_t> cardinalities = get_class_cardinalities(header, bcm);

      typename PForest<TestState, TestData>::TreeVec forest;
      forest.reserve(nbtrees);

      std::vector<std::unique_ptr<TrainState>> states_vec;
      states_vec.reserve(nbtrees);

      auto mk_task = [&bcmvec, &states_vec, &mutex, &forest, &data, this](size_t tree_index) {
        { // Lock protecting the printing
          std::lock_guard lock(mutex);
          std::cout << "Start tree " << tree_index << std::endl;
        }
        auto start = libtempo::utils::now();
        auto tree = this->tree_trainer->train(*states_vec[tree_index], data, bcmvec);
        auto delta = libtempo::utils::now() - start;
        {
          // Lock protecting the forest and out printing
          std::lock_guard lock(mutex);
          // --- Printing
          auto cf = std::cout.fill();
          std::cout << std::setfill('0');
          std::cout << std::setw(3) << tree_index + 1 << " / " << nbtrees << "   ";
          std::cout.fill(cf);
          std::cout << " timing: " << libtempo::utils::as_string(delta) << std::endl;
          // --- Add in the forest
          forest.push_back(std::move(tree));
        }
      };

      // Create the tasks per tree. Note that we clone the state.
      libtempo::utils::ParTasks p;
      for (size_t i = 0; i<nbtrees; ++i) {
        states_vec.push_back(state.forest_fork(i));
        p.push_task(mk_task, i);
      }

      p.execute(nbthread);

      return {
        std::move(states_vec),
        std::make_shared<PForest<TestState, TestData>>(std::move(forest), cardinalities)
      };

    }
  };

}