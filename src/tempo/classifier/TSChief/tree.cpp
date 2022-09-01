#include "tree.hpp"

namespace tempo::classifier::TSChief {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Result of a trained tree
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /// Given a testing state and testing data, do a prediction for the exemplar 'index'
  classifier::Result1 TreeNode::predict(TreeState& state, TreeData const& data, size_t index) const {
    if (node_kind==LEAF) {
      return as_leaf.splitter->predict(state, data, index);
    } else {
      size_t branch_idx = as_node.splitter->get_branch_index(state, data, index);
      const auto& branch = as_node.branches.at(branch_idx);
      return branch->predict(state, data, index);
    }
  }

  // --- --- --- Static functions

  std::unique_ptr<TreeNode> TreeNode::make_leaf(std::unique_ptr<i_SplitterLeaf> sleaf) {
    return std::unique_ptr<TreeNode>(
      new TreeNode{.node_kind = LEAF, .as_leaf = Leaf{std::move(sleaf)}}
    );
  }

  std::unique_ptr<TreeNode> TreeNode::make_node(std::unique_ptr<i_SplitterNode> snode,
                                                std::vector<BRANCH>&& branches) {
    return std::unique_ptr<TreeNode>(
      new TreeNode{.node_kind = NODE, .as_node = Node{std::move(snode), std::move(branches)}}
    );
  }


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Training a tree
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  std::unique_ptr<TreeNode> TreeTrainer::train(TreeState& state, const TreeData& data, ByClassMap const& bcm) const {
    // Ensure that we have at least one class reaching this node!
    // Note: there may be no data point associated to the class.
    assert(bcm.nb_classes()>0);

    // Try to generate a sleaf; if successful, make a sleaf node
    typename i_GenLeaf::Result opt_leaf = leaf_generator->generate(state, data, bcm);

    if (opt_leaf) {
      // --- --- --- LEAF
      return TreeNode::make_leaf(std::move(opt_leaf.value()));
    } else {
      // --- --- --- NODE
      // If we could not generate a sleaf, make a node.
      // Recursively build each branches, then build the current node
      i_GenNode::Result rnode = node_generator->generate(state, data, bcm);
      const size_t nb_branches = rnode.branch_splits.size();
      std::vector<TreeNode::BRANCH> branches;
      branches.reserve(nb_branches);

      // Building loop
      for (size_t idx = 0; idx<nb_branches; ++idx) {
        ByClassMap const& branch_bcm = rnode.branch_splits.at(idx);

        // Signal the state that we are going down a new branch
        state.start_branch(idx);

        // Build the branch
        std::unique_ptr<TreeNode> branch = train(state, data, branch_bcm);
        branches.push_back(std::move(branch));

        // Signal the state that we are done with this branch
        state.end_branch(idx);
      }

      // Result
      return TreeNode::make_node(std::move(rnode.splitter), std::move(branches));
    }
  } // End of train method

} // End of tempo::classifier::TSChief