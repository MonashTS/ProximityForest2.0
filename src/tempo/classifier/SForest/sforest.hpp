#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/dts.hpp>
#include <tempo/classifier/utils.hpp>

namespace tempo::classifier::SForest {

  // --- --- --- --- --- ---
  // State Concepts

  /// BaseState concept
  /// Ensure that a state can be forked/merged
  template<typename S>
  concept BaseState = std::movable<S>&&requires(S& s0){

    requires requires(size_t branch_idx){
      { s0.branch_fork(branch_idx) } -> std::same_as<S>;
    };

    requires requires(S&& s1){
      s0.branch_merge(std::move(s1));
    };

  };

  /// TreeState concept for the SForest.
  /// A state must (at least) offer a PRNG, and a couple of forking/merging methods
  template<typename S>
  concept TreeState = BaseState<S>&&requires(S& s) {
    s.prng;
  };

  // --- --- --- --- --- ---
  // Data Concepts

  /// Train Data concept: must give access to the training dataset
  template<typename D>
  concept TrainData = requires(D& d, std::string tname){
    { d.get_train_dataset(tname) } -> std::convertible_to<DTS>;
    { d.get_train_header() } -> std::convertible_to<DatasetHeader const&>;
  };

  /// Train Data concept: must give access to both the training dataset and the test dataset
  template<typename D>
  concept TestData = requires(D& d, std::string tname){
    { d.get_test_dataset(tname) } -> std::convertible_to<DTS>;
    { d.get_test_header() } -> std::convertible_to<DatasetHeader const&>;
  };

  // --- --- --- --- --- ---
  // Splitter Interfaces

  /// Test time leaf splitter interface
  template<TreeState TestS, typename TestD>
  struct LeafSplitter_i {

    /// Return type of a leaf. Transmit the state back and produces a classifier::Result
    struct R {
      std::unique_ptr<TestS> state;
      classifier::Result result;
    };

    /// Use the leaf splitter to predict a result using a state, the test data, and an index used to identify the
    /// test exemplar within the test data.
    virtual R predict(std::unique_ptr<TestS> state, TestD const& data, size_t index) = 0;

    virtual ~LeafSplitter_i() = default;
  };

  /// Test time node splitter interface
  template<TreeState TestS, typename TestD>
  struct NodeSplitter_i {

    /// Return type of a node. Transmit the state back and produces the index of the branch to follow.
    struct R {
      std::unique_ptr<TestS> state;
      size_t branch_index;
    };

    /// Use the node splitter to predict a result using a state, the test data, and an index used to identify the
    /// test exemplar within the test data.
    virtual R get_branch_index(std::unique_ptr<TestS> state, TestD const& data, size_t index) = 0;

    virtual ~NodeSplitter_i() = default;
  };

  // --- --- --- --- --- ---
  // Node/Tree implementation

  /// Splitting Node - splitting nodes are arranged to form a splitting tree.
  template<TreeState TestS, typename TestD>
  struct SNode {
    using LEAF = LeafSplitter_i<TestS, TestD>;
    using NODE = NodeSplitter_i<TestS, TestD>;
    using LEAF_SPLITTER = std::unique_ptr<LeafSplitter_i<TestS, TestD>>;
    using NODE_SPLITTER = std::unique_ptr<NodeSplitter_i<TestS, TestD>>;
    using BRANCH = std::unique_ptr<SNode<TestS, TestD>>;

    /// Sum Type index
    enum NodeKind { K_LEAF, K_NODE };

    /// Internal Node Type when node_kind == K_LEAF
    struct Leaf {
      LEAF_SPLITTER splitter;
    };

    /// Internal Node Type when node_kind == K_NODE
    struct Node {
      NODE_SPLITTER splitter;
      std::vector<BRANCH> branches;
    };

    /// Disjunction index
    NodeKind node_kind;

    /// When node_kind == K_LEAF
    Leaf as_leaf{};

    /// When node_kind == K_NODE
    Node as_node{};

    static std::unique_ptr<SNode> make_node(NODE_SPLITTER splitter, std::vector<BRANCH>&& branches) {
      return std::unique_ptr<SNode>(new SNode{
        .node_kind = K_NODE,
        .as_node = Node{std::move(splitter), std::move(branches)}
      });
    }

    static std::unique_ptr<SNode<TestS, TestD>> make_leaf(LEAF_SPLITTER splitter) {
      return std::unique_ptr<SNode>(new SNode{
        .node_kind = K_LEAF,
        .as_leaf = Leaf{std::move(splitter)}
      });
    }

    /// Classification result. Transmit the state back and produces a classifier::Result
    struct R {
      std::unique_ptr<TestS> state;
      classifier::Result result;

      explicit R(typename LEAF::R&& r) {
        state = std::move(r.state);
        result = std::move(r.result);
      }

    };

    /// Given a testing state and testing data, do a prediction for the exemplar 'index'
    R predict(std::unique_ptr<TestS> state0, TestD const& data, size_t index) {
      if (node_kind==K_LEAF) {
        return R(as_leaf.splitter->predict(std::move(state0), data, index));
      } else {
        typename NODE::R r = as_node.splitter->get_branch_index(std::move(state0), data, index);
        const auto& branch = as_node.branches.at(r.branch_index);
        return branch->predict(std::move(r.state), data, index);
      }
    }

  };


  // --- --- --- --- --- ---
  // Splitter Generator Interfaces

  /// Train time leaf splitter generator
  template<TreeState TrainS, typename TrainD, typename TestS, typename TestD>
  struct LeafSplitterGen_i {

    /// Return type of a leaf generator.
    /// Transmit back the train state, and optionally returns a leaf splitter.
    /// Returning no leaf splitter means that an node splitter must be built instead.
    struct R {
      std::unique_ptr<TrainS> state;
      std::optional<std::unique_ptr<LeafSplitter_i<TestS, TestD>>> o_splitter;
    };

    /// Given a training state, training data, and a set of index (in a ByClassMap), try to generate a leaf
    virtual R generate(std::unique_ptr<TrainS> state, TrainD const& data, ByClassMap const& bcm) = 0;

    virtual ~LeafSplitterGen_i() = default;
  };

  /// Train time node splitter generator
  template<TreeState TrainS, typename TrainD, typename TestS, typename TestD>
  struct NodeSplitterGen_i {

    /// Return type of a node generator.
    /// Transmit back the train state, and a node splitter.
    /// Also transmit the split of the incoming train data.
    /// The actual split; The size of the vector tells us the number of branches.
    /// For each branch, the associated ByClassMap can contain no index, but **cannot** contain no label.
    /// If a branch is required, but no train actually data reaches it,
    /// the BCM must contains at least one label mapping to an empty set of index.
    struct R {
      std::unique_ptr<TrainS> state;
      std::unique_ptr<NodeSplitter_i<TestS, TestD>> splitter;
      std::vector<ByClassMap> branch_splits;
    };

    /// Given a training state, training data, and a set of index (in a ByClassMap), try to generate a leaf
    virtual R generate(std::unique_ptr<TrainS> state, TrainD const& data, ByClassMap const& bcm) = 0;

    virtual ~NodeSplitterGen_i() = default;
  };


  // --- --- --- --- --- ---
  // Node/Tree trainer implementation

  /// Train a Splitting Tree
  template<TreeState TrainS, typename TrainD, typename TestS, typename TestD>
  struct STreeTrainer {
    // Shorthand for leaf and node generators types
    using GLeaf = LeafSplitterGen_i<TrainS, TrainD, TestS, TestD>;
    using GNode = NodeSplitterGen_i<TrainS, TrainD, TestS, TestD>;

    // Shorthand for result type
    using RLeaf = typename GLeaf::R;
    using RNode = typename GNode::R;

    // leaf and node generators; the usually themselves call other generators.
    std::shared_ptr<GLeaf> leaf_generator;
    std::shared_ptr<GNode> node_generator;

    /// Build a Splitting Tree with a leaf generator and a node generator
    STreeTrainer(std::shared_ptr<GLeaf> leaf_generator, std::shared_ptr<GNode> node_generator) :
      leaf_generator(leaf_generator), node_generator(node_generator) {}

    /// Result type of the training method.
    /// Transmit back the training state and the tree.
    struct R {
      std::unique_ptr<TrainS> state;
      std::unique_ptr<SNode<TestS, TestD>> tree;
    };

    /// Train a splitting tree
    R train(std::unique_ptr<TrainS> state0, const TrainD& data, ByClassMap const& bcm) {
      // Ensure that we have at least one class reaching this node!
      // Note: there may be no data point associated to the class.
      assert(bcm.nb_classes()>0);

      // Shorthands
      using STATE = std::unique_ptr<TrainS>;
      using TREE = SNode<TestS, TestD>;

      // Try to generate a leaf
      RLeaf rleaf = leaf_generator->generate(std::move(state0), data, bcm);
      STATE state1 = std::move(rleaf.state);

      if (rleaf.o_splitter) {
        // --- Make a leaf
        return R{
          std::move(state1),
          TREE::make_leaf(std::move(rleaf.o_splitter.value()))
        };
      } else {
        // --- Make a node
        RNode rnode = node_generator->generate(std::move(state1), data, bcm);
        STATE state2 = std::move(rnode.state);
        // Build the branches, then build the current node
        const size_t nb_branches = rnode.branch_splits.size();
        std::vector<typename TREE::BRANCH> branches;
        branches.reserve(nb_branches);
        // Building loop
        for (size_t idx = 0; idx<nb_branches; ++idx) {
          ByClassMap const& branch_bcm = rnode.branch_splits.at(idx);
          // Clone the state
          STATE branch_state0 = std::make_unique<TrainS>(state2->branch_fork(idx));
          R r = train(std::move(branch_state0), data, branch_bcm);
          branches.push_back(std::move(r.tree));
          // Merge state
          state2->branch_merge(std::move(*r.state));
        }
        // Result
        return R{
          std::move(state2),
          TREE::make_node(std::move(rnode.splitter), std::move(branches))
        };
      }

    } // End of train method

  }; // End of struct STreeTrainer

} // End of namespace tempo::classifier::SForest
