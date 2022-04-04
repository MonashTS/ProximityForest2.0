#pragma once

#include <libtempo/concepts.hpp>
#include <libtempo/tseries/dataset.hpp>

#include <functional>
#include <variant>

namespace libtempo::classifier::pf {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- Leaf
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /** Test Time Interface: trained leaf node - classifier */
  template<Label L, typename Stest>
  struct IPF_LeafSplitter {

    /// Predict the probability of a classes at the leaf. Order must respect label_to_index from DatasetHeader.
    virtual std::vector<double> predict_proba(Stest& state, size_t test_index) = 0;

    virtual ~IPF_LeafSplitter() = default;
  };

  /** Train time result type when generating a leaf */
  template<Label L, typename Stest>
  struct ResLeaf {
    /// Resulting splitter to use at test time
    std::unique_ptr<IPF_LeafSplitter<L, Stest>> splitter;
  };

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- Internal node
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /** Test Time Interface: trained internal node - split between branches */
  template<Label L, typename Stest>
  struct IPF_NodeSplitter {

    /// Get the branch index
    virtual size_t get_branch_index(Stest& state, size_t test_index) = 0;

    virtual ~IPF_NodeSplitter() = default;
  };

  /** Train time internal node split */
  template<Label L>
  struct PF_NodeSplit {
    /// The actual split: the size of the vector tells us the number of branches
    /// Note: a ByClassMap can contain no indexes, but cannot contain no label.
    ///       if a branch is required, but no train actually data reaches it,
    ///       the map contains labels (at least one) mapping to empty IndexSet
    std::vector<ByClassMap<L>> split;
  };

  /** Result type when generating an internal node */
  template<Label L, typename Stest>
  struct ResNode {

    /// Train time split, with an entry per sub-branch
    PF_NodeSplit<L> branch_splits;

    /// Resulting splitter to use at test time
    std::unique_ptr<IPF_NodeSplitter<L, Stest>> splitter;
  };

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- Generator
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /** Train time interface: a node generator.
   * @tparam L          Label type
   * @tparam Strain     Train time state type
   * @tparam Stest      Test time state type
   */
  template<Label L, typename Strain, typename Stest>
  struct IPF_NodeGenerator {
    /// Shorthand for result type: either a leaf or an internal node
    using Result = std::variant<ResLeaf<L, Stest>, ResNode<L, Stest>>;

    /** Generate a new splitter from a training state and the ByClassMap at the node.
    * @param state  Training state - mutable reference!
    * @param bcmvec stack of BCM from root to this node: 'bcmvec.back()' stands for the BCM at this node.
    * @return  ISplitterGenerator::Result with the splitter and the associated split of the train data
    */
    virtual Result generate(Strain& state, const std::vector<ByClassMap<L>>& bcm) const = 0;

    virtual ~IPF_NodeGenerator() = default;
  };

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- State
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /** Interface for the train state */
  template<Label L, typename Strain, typename Stest>
  struct IStrain {

    /// Callback when making a leaf - XOR with on_make_branches
    virtual void on_make_leaf(const ResLeaf<L, Stest>& /* leaf */){}

    /// Callback when making a node with the result from a splitter generator - XOR with on_make_leaf
    virtual void on_make_branches(const ResNode<L, Stest>& /* inode */) {}

    /// Clone the state for sub branch/sub tree with index "bidx", leading to a "sub state"
    virtual Strain clone(size_t /* bidx */) = 0;

    /// Merge in "this" the "substate".
    virtual void merge(Strain&& /* substate */) = 0;

    virtual ~IStrain() = default;
  };

}
