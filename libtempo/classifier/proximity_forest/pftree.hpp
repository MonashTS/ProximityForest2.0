#pragma once

#include <libtempo/concepts.hpp>
#include <libtempo/tseries/dataset.hpp>
#include <libtempo/utils/utils.hpp>
#include <libtempo/classifier/proximity_forest/ipf.hpp>

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
  template<Label L, typename Stest>
  struct PFTree {
    using LeafSplitter = std::unique_ptr<IPF_LeafSplitter<L, Stest>>;
    using InnerNodeSplitter = std::unique_ptr<IPF_NodeSplitter<L, Stest>>;
    using Branches = std::vector<std::unique_ptr<PFTree<L, Stest>>>;

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
     * @param query_idx     Index of the exemplar in the test dataset
     * @return A vector of probabilities. See DatasetHeader.
     */
    [[nodiscard]]
    static std::vector<double> predict_proba(const PFTree& pt, Stest& state, size_t query_idx) {
      switch (pt.node.index()) {
      case 0: { return std::get<0>(pt.node)->predict_proba(state, query_idx); }
      case 1: {
        const InnerNode& n = std::get<1>(pt.node);
        size_t branch_idx = n.splitter->get_branch_index(state, query_idx);
        const auto& sub = n.branches.at(branch_idx);    // Use at because it is 'const', when [ ] is not
        return predict_proba(*sub, state, query_idx);
      }
      default: utils::should_not_happen();
      }
    }

  public:

    /** Classification function
     * @param state     Test time state. Must give access to the dataset and other info required by the splitters
     * @param index     Index of the query in the test dataset
     * @return  Vector of probability. See DatasetHeader.
     */
    [[nodiscard]]
    std::vector<double> predict_proba(Stest& state, size_t index) const {
      return predict_proba(*this, state, index);
    }

  }; // End of PTree


  /** Train time Proximity tree */
  template<Label L, typename Strain, typename Stest>
  struct PFTreeTrainer {

    // Shorthand for result type
    using Result = std::variant<ResLeaf<L, Strain, Stest>, ResNode<L, Strain, Stest>>;
    using LeafResult = typename IPF_LeafGenerator<L, Strain, Stest>::Result;
    using NodeResult = typename IPF_NodeGenerator<L, Strain, Stest>::Result;
    using BCMVec = std::vector<ByClassMap<L>>;

    std::shared_ptr<IPF_LeafGenerator<L, Strain, Stest>> leaf_generator;
    std::shared_ptr<IPF_NodeGenerator<L, Strain, Stest>> node_generator;

    /// Build a proximity tree trainer with a leaf generator and a node generator
    PFTreeTrainer(
      std::shared_ptr<IPF_LeafGenerator<L, Strain, Stest>>
      leaf_generator,
      std::shared_ptr<IPF_NodeGenerator<L, Strain, Stest>> node_generator) :
      leaf_generator(leaf_generator), node_generator(node_generator) {}

  private:

    /** Generate a new splitter from a training state and the ByClassMap at the node.
    * @param state  Training state - mutable reference!
    * @param bcmvec stack of BCM from root to this node: 'bcmvec.back()' stands for the BCM at this node.
    * @return  ISplitterGenerator::Result with the splitter and the associated split of the train data
    */
    Result generate(Strain& state, const BCMVec& bcmvec) const {
      LeafResult oleaf = leaf_generator->generate(state, bcmvec);
      if (oleaf.has_value()) {
        return Result{std::move(oleaf.value())};
      } else {
        return Result{node_generator->generate(state, bcmvec)};
      }
    }

  public:

    /// Train a tree
    [[nodiscard]]
    std::unique_ptr<PFTree<L, Stest>> train(Strain& strain, BCMVec bcmvec)
    const requires std::derived_from<Strain, IStrain<L, Strain, Stest>> {
      // Ensure that we have at least one class reaching this node!
      // Note: there may be no data point associated to the class.
      const auto bcm = bcmvec.back();
      assert(bcm.nb_classes()>0);
      // Final return 'ret' variable
      std::unique_ptr<PFTree<L, Stest>> ret;

      // Call the generator and analyse the result
      Result result = generate(strain, bcmvec);

      switch (result.index()) {
      case 0: { // Leaf case - stop the recursion, build a leaf node
        ResLeaf<L, Strain, Stest> leaf = std::get<0>(std::move(result));
        // Train state callback
        leaf.callback(strain);
        // Build the leaf node with the splitter
        ret = std::unique_ptr<PFTree<L, Stest>>(new PFTree<L, Stest>{.node=std::move(leaf.splitter)});
        break;
      }
      case 1: { // Inner node case: recursion per branch
        ResNode<L, Strain, Stest> inner_node = std::get<1>(std::move(result));
        // Train state callback
        inner_node.callback(strain);
        // Build subbranches, then we'll build the current node
        const size_t nbbranches = inner_node.branch_splits.size();
        typename PFTree<L, Stest>::Branches subbranches;
        subbranches.reserve(nbbranches);
        for (size_t idx = 0; idx<nbbranches; ++idx) {
          // Clone state, push bcm
          bcmvec.template emplace_back(std::move(inner_node.branch_splits[idx]));
          Strain sub_state = strain.branch_fork(idx);
          // Sub branch
          subbranches.push_back(train(sub_state, bcmvec));
          // Merge state, pop bcm
          strain.branch_merge(std::move(sub_state));
          bcmvec.pop_back();
        }
        // Now that we have all the subbranches, build the current node
        ret = std::unique_ptr<PFTree<L, Stest>>
          (new PFTree<L, Stest>{.node= typename PFTree<L, Stest>::InnerNode{
             .splitter = std::move(inner_node.splitter),
             .branches = std::move(subbranches)
           }}
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
  template<Label L, typename Stest>
  struct PForest {
    using TreeVec = std::vector<std::unique_ptr<PFTree<L, Stest>>>;

    TreeVec forest;
    size_t nb_classes;

    PForest(TreeVec&& forest, size_t nb_classes) :
      forest(std::move(forest)), nb_classes(nb_classes) {}

    [[nodiscard]]
    std::vector<double> predict_proba(Stest& state, size_t index) const {

      const size_t nbtree = forest.size();
      std::vector<utils::StddevWelford> result_welford(nb_classes);

      for (size_t i = 0; i<nbtree; ++i) {
        auto proba = forest[i]->predict_proba(state, index);
        // Accumulate probabilities
        for (size_t j = 0; j<nb_classes; ++j) { result_welford[j].update(proba[j]); }
      }

      std::vector<double> result(nb_classes);
      for (size_t j = 0; j<nb_classes; ++j) { result[j] = result_welford[j].get_mean(); }

      return result;
    }
  };

  /** Proximity Forest at train time */
  template<Label L, typename Strain, typename Stest>
  struct PForestTrainer {
    using BCMVec = std::vector<ByClassMap<L>>;

    std::shared_ptr<const PFTreeTrainer<L, Strain, Stest>> tree_trainer;
    size_t nbtrees;

    explicit PForestTrainer(std::shared_ptr<PFTreeTrainer<L, Strain, Stest>> tree_trainer, size_t nbtrees) :
      tree_trainer(std::move(tree_trainer)),
      nbtrees(nbtrees) {}

    /** Train the proximity forest */
    std::unique_ptr<PForest<L, Stest>> train(Strain& state, BCMVec bcmvec) {

      std::mutex mutex;
      typename PForest<L, Stest>::TreeVec forest;
      forest.reserve(nbtrees);

      std::vector<std::unique_ptr<Strain>> states_vec;
      states_vec.reserve(nbtrees);

      auto mk_task = [&bcmvec, &states_vec, &mutex, &forest, this](size_t tree_index) {
        auto tree = this->tree_trainer->train(*states_vec[tree_index], bcmvec);
        // Start lock: protecting the forest and out printing
        std::lock_guard lock(mutex);
        // --- Printing
        auto cf = std::cout.fill();
        std::cout << std::setfill('0');
        std::cout << std::setw(3) << tree_index + 1 << " / " << nbtrees << "   ";
        std::cout.fill(cf);
        std::cout << std::endl;
        // --- Add in the forest
        forest.push_back(std::move(tree));
      };

      tempo::ParTasks p;
      for (size_t i = 0; i<nbtrees; ++i) {
        states_vec.push_back(state.forest_clone());
        p.push_task(mk_task, i);
      }
      p.execute(8);

      for (size_t i = 0; i<nbtrees; ++i) {
        state.forest_merge(std::move(states_vec[i]));
      }


      // for (size_t i = 0; i<nbtrees; ++i) {
      //   auto clone = state.forest_clone();
      //   BCMVec copy = bcmvec;
      //   forest.push_back(tree_trainer->train(*clone, copy));
      //   state.forest_merge(std::move(clone));
      // }

      size_t nb_classes = state.get_header().nb_labels();

      return std::make_unique<PForest<L, Stest>>(std::move(forest), nb_classes);

    }
  };

}