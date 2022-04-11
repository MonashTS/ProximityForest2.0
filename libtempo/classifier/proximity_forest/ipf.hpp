#pragma once

#include <libtempo/concepts.hpp>
#include <libtempo/tseries/dataset.hpp>
#include <libtempo/utils/utils.hpp>

#include <functional>
#include <variant>

namespace libtempo::classifier::pf {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- Leaf
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /** Test Time Interface: trained leaf node - classifier */
  template<Label L, typename Stest>
  struct IPF_LeafSplitter {

    /** Predict the probability of a classes at the leaf.
     *  Order must respect the 'label_to_index' order from DatasetHeader.
     *  The 1st component of the tuple is an associated weight used when ensembling; Set it to 1.0 to ignore its effect.
     */
    virtual std::tuple<double, std::vector<double>> predict_proba(Stest& state, size_t test_index) const = 0;

    virtual ~IPF_LeafSplitter() = default;
  };

  /** Train time result type when generating a leaf */
  template<Label L, typename Strain, typename Stest>
  struct ResLeaf {

    /// Resulting splitter to use at test time
    std::unique_ptr<IPF_LeafSplitter<L, Stest>> splitter;

    /// Function used to update the state with statistics
    std::function<void(Strain& state)> callback = [](Strain& /* state */ ){};

  };

  /** Train time interface: a leaf generator.
   * @tparam L          Label type
   * @tparam Strain     Train time state type
   * @tparam Stest      Test time state type
   */
  template<Label L, typename Strain, typename Stest>
  struct IPF_LeafGenerator {
    using Result = std::optional<ResLeaf<L, Strain, Stest>>;

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
    virtual size_t get_branch_index(Stest& state, size_t test_index) const = 0;

    virtual ~IPF_NodeSplitter() = default;
  };

  /** Result type when generating an internal node */
  template<Label L, typename Strain, typename Stest>
  struct ResNode {

    /// The actual split: the size of the vector tells us the number of branches
    /// Note: a ByClassMap can contain no indexes, but cannot contain no label.
    ///       if a branch is required, but no train actually data reaches it,
    ///       the map contains labels (at least one) mapping to empty IndexSet
    std::vector<ByClassMap<L>> branch_splits;

    /// Resulting splitter to use at test time
    std::unique_ptr<IPF_NodeSplitter<L, Stest>> splitter;

    /// Function used to update the state with statistics
    std::function<void(Strain& state)> callback = [](Strain& /* state */ ){};
  };

  /** Train time interface: a node generator. */
  template<Label L, typename Strain, typename Stest>
  struct IPF_NodeGenerator {
    using Result = ResNode<L, Strain, Stest>;

    /** Generate a new splitter from a training state and the ByClassMap at the node. */
    virtual Result generate(Strain& state, const std::vector<ByClassMap<L>>& bcm) const = 0;

    virtual ~IPF_NodeGenerator() = default;
  };


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- State
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /// Requires a type to have a ".prng" field which is a random number generator
  template<typename T>
  concept has_prng = requires {
    std::uniform_random_bit_generator<decltype(*T::prng)>;
  };

  /// Description of a dataset map
  template<Float F, Label L>
  using DatasetMap_t = std::map<std::string, libtempo::DTS<F, L>>;

  /// Requirement for a shared map
  template<typename T, typename F, typename L>
  concept TimeSeriesDataset = requires {
    std::same_as<decltype(T::dataset_shared_map), std::shared_ptr<Dataset<F, L>>>;
  };

  /// Dataset Header mixin: provides a get_header() providing an non empty dataset_shared_map
  template<typename Base, Float F, Label L>
  struct TimeSeriesDatasetHeader {
    const DatasetHeader<L>& get_header() const {
      const auto& ds = static_cast<const Base&>(*this);
      const std::map<std::string, libtempo::DTS<F, L>>& map = *ds.dataset_shared_map;
      return map.begin()->second.header();
    }
  };

  /** Interface for the train state */
  template<Label L, typename Strain, typename Stest>
  struct IStrain {

    /// Fork the state for sub branch/sub tree with index "bidx", leading to a "sub state"
    virtual Strain branch_fork(size_t /* bidx */) = 0;

    /// Merge-move "other" into "this".
    virtual void branch_merge(Strain&& /* other */) = 0;

    /// Clone at the forest level - clones must be fully independent as they can be used in parallel
    virtual std::unique_ptr<Strain> forest_clone() = 0;

    /// Merge in this a state that has been produced by forest_clone
    virtual void forest_merge(std::unique_ptr<Strain> other) = 0;

    virtual ~IStrain() = default;
  };

}
