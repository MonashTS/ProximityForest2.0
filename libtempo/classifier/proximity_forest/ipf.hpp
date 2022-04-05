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

  /** Train time interface: a leaf generator.
   * @tparam L          Label type
   * @tparam Strain     Train time state type
   * @tparam Stest      Test time state type
   */
  template<Label L, typename Strain, typename Stest>
  struct IPF_LeafGenerator {
    using Result = std::optional<ResLeaf<L, Stest>>;

    /** Generate a leaf from a training state and the ByClassMap at the node.
     *  If no leaf is to be generated, return the empty option, which will trigger the call of a NodeGenerator */
    virtual Result generate(Strain& state, const std::vector<ByClassMap<L>> & bcm) const = 0;

    virtual ~IPF_LeafGenerator() = default;
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

  /** Result type when generating an internal node */
  template<Label L, typename Stest>
  struct ResNode {

    /// The actual split: the size of the vector tells us the number of branches
    /// Note: a ByClassMap can contain no indexes, but cannot contain no label.
    ///       if a branch is required, but no train actually data reaches it,
    ///       the map contains labels (at least one) mapping to empty IndexSet
    std::vector<ByClassMap<L>> branch_splits;

    /// Resulting splitter to use at test time
    std::unique_ptr<IPF_NodeSplitter<L, Stest>> splitter;
  };

  /** Train time interface: a node generator. */
  template<Label L, typename Strain, typename Stest>
  struct IPF_NodeGenerator {
    using Result = ResNode<L, Stest>;

    /** Generate a new splitter from a training state and the ByClassMap at the node. */
    virtual Result generate(Strain& state, const std::vector<ByClassMap<L>>& bcm) const = 0;

    virtual ~IPF_NodeGenerator() = default;
  };

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- Main node generator class
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /** Train time node generator. */
  template<Label L, typename Strain, typename Stest>
  struct PF_TopGenerator {
    // Shorthand for result type
    using Result = std::variant<ResLeaf<L, Stest>, ResNode<L, Stest>>;
    using LeafResult = typename IPF_LeafGenerator<L, Strain, Stest>::Result;
    using NodeResult = typename IPF_NodeGenerator<L, Strain, Stest>::Result;

    std::shared_ptr<IPF_LeafGenerator<L, Strain, Stest>> leaf_generator;
    std::shared_ptr<IPF_NodeGenerator<L, Strain, Stest>> node_generator;

    PF_TopGenerator(
      std::shared_ptr<IPF_LeafGenerator<L, Strain, Stest>> leaf_generator,
      std::shared_ptr<IPF_NodeGenerator<L, Strain, Stest>> node_generator
    ) : leaf_generator(leaf_generator), node_generator(node_generator) {}

    /** Generate a new splitter from a training state and the ByClassMap at the node.
    * @param state  Training state - mutable reference!
    * @param bcmvec stack of BCM from root to this node: 'bcmvec.back()' stands for the BCM at this node.
    * @return  ISplitterGenerator::Result with the splitter and the associated split of the train data
    */
    Result generate(Strain& state, const std::vector<ByClassMap<L>>& bcmvec) const {
      LeafResult oleaf = leaf_generator->generate(state, bcmvec);
      if (oleaf.has_value()) {
        return Result{std::move(oleaf.value())};
      } else {
        return Result{node_generator->generate(state, bcmvec)};
      }
    }
  };


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- State
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /** Interface for the train state */
  template<Label L, typename Strain, typename Stest>
  struct IStrain {

    /// Callback when making a leaf - XOR with on_make_branches
    virtual void on_make_leaf(const ResLeaf<L, Stest>& /* leaf */) {}

    /// Callback when making a node with the result from a splitter generator - XOR with on_make_leaf
    virtual void on_make_branches(const ResNode<L, Stest>& /* inode */) {}

    /// Clone the state for sub branch/sub tree with index "bidx", leading to a "sub state"
    virtual Strain clone(size_t /* bidx */) = 0;

    /// Merge in "this" the "substate".
    virtual void merge(Strain&& /* substate */) = 0;

    virtual ~IStrain() = default;
  };

}
