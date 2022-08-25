#pragma once

#include <any>
#include <memory>
#include <vector>
#include "tempo/classifier/utils.hpp"

namespace tempo::classifier::sf {

  struct i_TreeState {
    // Virtual destructor
    virtual ~i_TreeState() = default;

    // --- --- --- Fork/Merge
    virtual std::unique_ptr<i_TreeState> forest_fork() const = 0;
    virtual void forest_merge_in(std::unique_ptr<i_TreeState>&& other) = 0;

    // --- --- --- Start/End node

    /// Method called when a new branch is started - will be called before calling the "train" function for this branch.
    /// Branches are created in a "deep first" fashion.
    virtual void start_branch(size_t branch_idx) = 0;

    /// Method called when a branch is done - will be called after calling the "train" function for this branch.
    virtual void end_branch(size_t branch_idx) = 0;
  };

  struct TreeState : public i_TreeState {
    std::vector<std::unique_ptr<i_TreeState>> states{};

    inline size_t register_state(std::unique_ptr<i_TreeState>&& s){
      size_t idx = states.size();
      states.push_back(std::move(s));
      return idx;
    }

    i_TreeState& at(size_t idx){

    }


    inline std::unique_ptr<i_TreeState> forest_fork() const override {
      // Create the other state and for substates 1 for 1
      auto fork = std::make_unique<TreeState>();
      for (auto const& substate : states) { fork->states.push_back(substate->forest_fork()); }
      return std::move(fork);
    }

    inline void forest_merge_in(std::unique_ptr<i_TreeState>&& other) override {
      // Get pointer of the good type
      auto *other_state = dynamic_cast<TreeState *>(other.get());
      if (other_state==nullptr) { tempo::utils::should_not_happen("Dynamic cast to TreeState failed"); }
      // Merge in the substates in a 1 to 1 index matching
      for (size_t i{0}; i<states.size(); ++i) {
        auto&& upt = std::move(other_state->states[i]);
        states[i]->forest_merge_in(std::move(upt));
      }
    }

    inline void start_branch(size_t branch_idx) override {
      for (auto& substate : states) { substate->start_branch(branch_idx); }
    }

    inline void end_branch(size_t branch_idx) override {
      for (auto& substate : states) { substate->end_branch(branch_idx); }
    }

  };

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  template<typename TestD>
  struct i_SplitterLeaf {
    // Virtual destructor
    virtual ~i_SplitterLeaf() = default;

    /// The Leaf Splitter predicts a result using a (mutable) state, the test data,
    /// and an index used to identify the test exemplar within the test data.
    virtual classifier::Result1 predict(TreeState& state, TestD const& data, size_t index) = 0;
  };

  template<typename TestD>
  struct i_SplitterNode {
    // Virtual destructor
    virtual ~i_SplitterNode() = default;

    /// The Node Splitter finds which branch to follow using a (mutable) state, the test data,
    /// and an index used to identify the test exemplar within the test data.
    virtual size_t get_branch_index(TreeState& state, TestD const& data, size_t index) = 0;
  };

  template<typename TestD>
  struct TreeNode {
    // --- --- --- Types
    using iLeaf = i_SplitterLeaf<TestD>;
    using iNode = i_SplitterNode<TestD>;
    using BRANCH = std::unique_ptr<TreeNode<TestD>>;

    /// Node Kind type: a TreeNode is either a leaf or an internal node
    enum Kind { LEAF, NODE };

    /// Payload type when node_kind == LEAF
    struct Leaf {
      std::unique_ptr<iLeaf> splitter;
    };

    /// Payload type when node_kind == NODE
    struct Node {
      std::unique_ptr<iNode> splitter;
      std::vector<BRANCH> branches;
    };

    // --- --- --- Fields

    Kind node_kind;
    Leaf as_leaf{};
    Node as_node{};

    // --- --- --- Methods

    /// Given a testing state and testing data, do a prediction for the exemplar 'index'
    classifier::Result1 predict(TreeState& state, TestD const& data, size_t index) {
      if (node_kind==LEAF) {
        return as_leaf.splitter->predict(state, data, index);
      } else {
        size_t branch_idx = as_node.splitter->get_branch_index(state, data, index);
        const auto& branch = as_node.branches.at(branch_idx);
        return branch->predict(state, data, index);
      }
    }

    // --- --- --- Static functions

    static std::unique_ptr<TreeNode> make_node(std::unique_ptr<iNode> splitter, std::vector<BRANCH>&& branches) {
      return std::unique_ptr<TreeNode>(
        new TreeNode{.node_kind = NODE, .as_node = Node{std::move(splitter), std::move(branches)}}
      );
    }

    static std::unique_ptr<TreeNode> make_leaf(std::unique_ptr<iLeaf> splitter) {
      return std::unique_ptr<TreeNode>(
        new TreeNode{.node_kind = LEAF, .as_leaf = Leaf{std::move(splitter)}}
      );
    }

  };

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  template<typename TrainD, typename TestD>
  struct i_GenLeaf {
    // --- --- --- Types
    using Result = typename std::optional<std::unique_ptr<i_SplitterLeaf<TestD>>>;

    // --- --- --- Constructor/Destructor
    virtual ~i_GenLeaf() = default;

    // --- --- --- Methods
    /// Given a training state, training data, and a set of index (in a ByClassMap), try to generate a leaf
    virtual Result generate(TreeState& state, TrainD const& data, ByClassMap const& bcm) = 0;
  };

  template<typename TrainD, typename TestD>
  struct i_GenNode {

    // --- --- --- Types

    /// Return type of a node generator.
    /// Return a splitter and the split of the incoming train data according to that splitter.
    /// The size of the branch_splits vector tells us the number of branches.
    /// For each branch, the associated ByClassMap can contain no index, but **cannot** contain no label.
    /// If a branch is required, but no train actually data reaches it,
    /// the BCM must contains at least one label mapping to an empty set of index.
    struct Result {
      std::unique_ptr<i_SplitterNode<TestD>> splitter;
      std::vector<ByClassMap> branch_splits;
    };

    // --- --- --- Constructor/Destructor

    virtual ~i_GenNode() = default;

    // --- --- --- Methods

    /// Given a training state, training data, and a set of index (in a ByClassMap), try to generate a leaf
    virtual Result generate(TreeState& state, TrainD const& data, ByClassMap const& bcm) = 0;

  };

  template<typename TrainD, typename TestD>
  struct TreeTrainer {
    // --- --- --- Types
    // Shorthand for leaf and node generators types
    using iGLeaf = i_GenLeaf<TrainD, TestD>;
    using iGNode = i_GenNode<TrainD, TestD>;

    // --- --- --- Fields
    // leaf and node generators
    std::shared_ptr<iGLeaf> leaf_generator;
    std::shared_ptr<iGNode> node_generator;

    // --- --- --- Constructors/Destructors

    /// Build a Splitting Tree with a leaf generator and a node generator
    TreeTrainer(std::shared_ptr<iGLeaf> leaf_generator, std::shared_ptr<iGNode> node_generator) :
      leaf_generator(leaf_generator), node_generator(node_generator) {}

    // --- --- --- Methods

    /// Train a splitting tree
    std::unique_ptr<TreeNode<TestD>> train(TreeState& state, const TrainD& data, ByClassMap const& bcm) const {
      // Ensure that we have at least one class reaching this node!
      // Note: there may be no data point associated to the class.
      assert(bcm.nb_classes()>0);

      // Try to generate a leaf; if successful, make a leaf node
      typename iGLeaf::Result rleaf = leaf_generator->generate(state, data, bcm);
      if (rleaf.o_splitter) {
        return TreeNode<TestD>::make_leaf(std::move(rleaf.o_splitter.value()));
      } else {
        // If we could not generate a leaf, make a node.
        // Recursively build each branches, then build the current node
        typename TreeNode<TestD>::Result rnode = node_generator->generate(state, data, bcm);
        const size_t nb_branches = rnode.branch_splits.size();
        std::vector<typename TreeNode<TestD>::BRANCH> branches;
        branches.reserve(nb_branches);
        // Building loop
        for (size_t idx = 0; idx<nb_branches; ++idx) {
          ByClassMap const& branch_bcm = rnode.branch_splits.at(idx);
          // Signal the state that we are going down a new branch
          state.start_branch(idx);
          // Build the branch
          std::unique_ptr<TreeNode<TestD>> branch = train(state, data, branch_bcm);
          branches.push_back(std::move(branch));
          // Signal the state that we are done with this branch
          state.end_branch(idx);
        }
        // Result
        return TreeNode<TestD>::make_node(std::move(rnode.splitter), std::move(branches));
      }

    } // End of train method

  };

} // End of tempo::classifier::sf