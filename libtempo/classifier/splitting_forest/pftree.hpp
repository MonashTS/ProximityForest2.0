#pragma once

#include <libtempo/concepts.hpp>
#include <libtempo/tseries/dataset.hpp>
#include <libtempo/utils/utils.hpp>
#include <libtempo/classifier/splitting_forest/ipf.hpp>

#include <functional>
#include <map>
#include <string>
#include <variant>

namespace libtempo::classifier::pf {

  /** Proximity Tree at test time
   *  Use an instance of PFTreeTrainer to obtain a trained PFTree, which can then be used to classify instances.
   *  Note: This class implements a node. A tree is represented by the root node.
   * @tparam L         Label type
   * @tparam Stest     Test State type. Must contain the info required by the splitters.
   */
  template<Label L, typename TestState, typename TestData>
  struct PFTree {
    using LeafSplitter = std::unique_ptr<IPF_LeafSplitter<L, TestState, TestData>>;
    using InnerNodeSplitter = std::unique_ptr<IPF_NodeSplitter<L, TestState, TestData>>;
    using Branches = std::vector<std::unique_ptr<PFTree<L, TestState, TestData>>>;

    struct InnerNode {
      InnerNodeSplitter splitter;
      Branches branches;
    };

    /** Leaf Node xor Inner Node */
    std::variant<LeafSplitter, InnerNode> node;

  private:

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

  }; // End of PTree


  /** Train time Proximity tree */
  template<Label L, typename TrainState, typename TrainData, typename TestState, typename TestData>
  struct PFTreeTrainer {

    // Shorthand for result type
    using Leaf = ResLeaf<L, TestState, TestData>;
    using Node = ResNode<L, TestState, TestData>;
    using Result = std::variant<Leaf, Node>;

    // BCM as vector
    using BCMVec = std::vector<ByClassMap<L>>;

    // Shorthands for Lead and Node generator types
    using LGen = IPF_LeafGenerator<L, TrainState, TrainData, TestState, TestData>;
    using NGen = IPF_NodeGenerator<L, TrainState, TrainData, TestState, TestData>;

    // Shorthand for the final tree type
    using PFTree_t = PFTree<L, TestState, TestData>;

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
        // State callback
        state.on_leaf(bcmvec);
        // Build the leaf node with the splitter
        ret = std::unique_ptr<PFTree_t>(new PFTree_t{.node=std::move(leaf.splitter)});
        break;
      }
      case 1: { // Inner node case: recursion per branch
        Node inner_node = std::get<1>(std::move(result));
        // Build subbranches, then we'll build the current node
        const size_t nbbranches = inner_node.branch_splits.size();
        typename PFTree_t::Branches subbranches;
        subbranches.reserve(nbbranches);
        for (size_t idx = 0; idx<nbbranches; ++idx) {
          // Clone state, push bcm
          bcmvec.template emplace_back(std::move(inner_node.branch_splits[idx]));
          TrainState sub_state = state.branch_fork(idx);
          // Sub branch
          subbranches.push_back(train(sub_state, data, bcmvec));
          // Merge state, pop bcm
          state.branch_merge(std::move(sub_state));
          bcmvec.pop_back();
        }
        // Now that we have all the subbranches, build the current node
        ret = std::unique_ptr<PFTree_t>(
          new PFTree_t{
            .node = typename PFTree_t::InnerNode{
              .splitter = std::move(inner_node.splitter),
              .branches = std::move(subbranches)
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
  template<Label L, typename TestState, typename TestData>
  struct PForest {
    using TreeVec = std::vector<std::unique_ptr<PFTree<L, TestState, TestData>>>;

    TreeVec forest;
    size_t nb_classes;

    PForest(TreeVec&& forest, size_t nb_classes) :
      forest(std::move(forest)), nb_classes(nb_classes) {}

    [[nodiscard]]
    std::tuple<double, std::vector<double>>
    predict_proba(TestState& state, const TestData& data, size_t instance_index, size_t nbthread) const {
      const size_t nbtree = forest.size();

      // Result variables
      std::vector<double> result(nb_classes);
      double total_weight = 0;

      // State vector
      std::vector<std::unique_ptr<TestState>> states_vec;
      states_vec.reserve(nbtree);

      // Multithreading control
      std::mutex mutex;

      auto test_task = [&](size_t tree_index) {
        auto [weight, proba] = forest[tree_index]->predict_proba(state, data, instance_index);
        { // Accumulate weighted probabilities
          std::lock_guard lock(mutex);
          for (size_t j = 0; j<nb_classes; ++j) { result[j] += weight*proba[j]; }
          total_weight += weight;
        }
      };

      // Create the tasks per tree. Note that we clone the state.
      libtempo::utils::ParTasks p;
      for (size_t i = 0; i<nbtree; ++i) {
        states_vec.push_back(state.forest_fork(i));
        p.push_task(test_task, i);
      }

      p.execute(nbthread);

      // Final divisions
      for (size_t j = 0; j<nb_classes; ++j) { result[j] /= total_weight; }

      return {total_weight, result};
    }
  };

  /** Proximity Forest at train time */
  template<Label L, typename TrainState, typename TrainData, typename TestState, typename TestData>
  struct PForestTrainer {
    using BCMVec = std::vector<ByClassMap<L>>;

    std::shared_ptr<const PFTreeTrainer<L, TrainState, TrainData, TestState, TestData>> tree_trainer;
    size_t nbtrees;

    PForestTrainer(
      std::shared_ptr<PFTreeTrainer<L, TrainState, TrainData, TestState, TestData>> tree_trainer,
      size_t nbtrees
    ) :
      tree_trainer(std::move(tree_trainer)),
      nbtrees(nbtrees) {}

    /** Train the proximity forest */
    std::tuple<
      std::vector<std::unique_ptr<TrainState>>,
      std::shared_ptr<PForest<L, TestState, TestData>>
    > train(TrainState& state, const TrainData& data, BCMVec bcmvec, size_t nbthread) {

      std::mutex mutex;
      typename PForest<L, TestState, TestData>::TreeVec forest;
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
      size_t nb_classes = data.get_header().nb_labels();

      return {
        std::move(states_vec),
        std::make_shared<PForest<L, TestState, TestData>>(std::move(forest), nb_classes)
      };

    }
  };

}