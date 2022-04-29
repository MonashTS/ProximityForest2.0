#pragma once

#include <libtempo/concepts.hpp>
#include <libtempo/tseries/dataset.hpp>
#include <libtempo/utils/utils.hpp>

#include <functional>
#include <variant>

namespace libtempo::classifier::pf {

  /// Shorthand for a vector of ByClassMap
  using BCMVec = std::vector<ByClassMap>;

  /// Result callback: when a node (leaf or internal) is generated, a result is produce.
  /// This result have a callback of the following type, allowing it to update the state in an arbitrary way.
  template<typename TrainState, typename TrainData>
  using CallBack = std::function<void(TrainState& state, const TrainData& data, const BCMVec& bcmvec)>;

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- Leaf
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /// Test Time Interface: trained leaf node - classifier
  template<typename TestState, typename TestData>
  struct IPF_LeafSplitter {

    /** Give the class cardinality per class at the leaf.
     *  Order must respect the 'label_to_index' order from DatasetHeader.
     */
     virtual arma::Col<size_t>
     predict_cardinality(TestState& state, const TestData& data, size_t test_index) const = 0;

    virtual ~IPF_LeafSplitter() = default;
  };

  /// Train time result type when generating a leaf
  template<typename TrainState, typename TrainData, typename TestState, typename TestData>
  struct ResLeaf {

    /// Resulting splitter to use at test time
    std::unique_ptr<IPF_LeafSplitter<TestState, TestData>> splitter;

    /// Callback
    CallBack<TrainState, TrainData> callback =
      [](TrainState& /* state */ , const TrainData& /* data */, const BCMVec& /* bcmvec */){};
  };

  /// Train time interface: a leaf generator.
  template<typename TrainState, typename TrainData, typename TestState, typename TestData>
  struct IPF_LeafGenerator {
    using Result = std::optional<ResLeaf<TrainState, TrainData, TestState, TestData>>;

    /// Generate a leaf from a training state and the ByClassMap at the node.
    /// If no leaf is to be generated, return the empty option, which will trigger the call of a NodeGenerator */
    virtual Result generate(TrainState& state , const TrainData& data, const BCMVec& bcm) const = 0;

    virtual ~IPF_LeafGenerator() = default;
  };


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- Internal node
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /// Test Time Interface: trained internal node - split between branches
  template<typename TestState, typename TestData>
  struct IPF_NodeSplitter {

    /// Get the branch index
    virtual size_t get_branch_index(TestState& state, const TestData& data, size_t test_index) const = 0;

    virtual ~IPF_NodeSplitter() = default;
  };

  /// Result type when generating an internal node
  template<typename TrainState, typename TrainData, typename TestState, typename TestData>
  struct ResNode {

    /// The actual split: the size of the vector tells us the number of branches
    /// Note: a ByClassMap can contain no indexes, but cannot contain no label.
    ///       if a branch is required, but no train actually data reaches it,
    ///       the map contains labels (at least one) mapping to empty IndexSet
    std::vector<ByClassMap> branch_splits;

    /// Resulting splitter to use at test time
    std::unique_ptr<IPF_NodeSplitter<TestState, TestData>> splitter;

    /// Callback
    CallBack<TrainState, TrainData> callback =
      [](TrainState& /* state */, const TrainData& /* data */, const BCMVec& /* bcmvec */){};

  };

  /// Train time interface: a node generator.
  template<typename TrainState, typename TrainData, typename TestState, typename TestData>
  struct IPF_NodeGenerator {
    using Result = ResNode<TrainState, TrainData, TestState, TestData>;

    /// Generate a new splitter from a training state and the ByClassMap at the node.
    virtual Result generate(TrainState& state, const TrainData& data, const BCMVec& bcm) const = 0;

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
  template<Float F>
  using DatasetMap_t = std::map<std::string, libtempo::DTS<F>>;

  /*
  /// Requirement for a shared map
  template<typename T, typename F, typename L>
  concept TimeSeriesDataset = requires {
    std::convertible_to<decltype(T::dataset_shared_map), std::shared_ptr<Dataset<F, L>>>;
  };

  /// Dataset Header mixin: provides a get_header() providing a non empty dataset_shared_map
  template<typename Base, Float F, Label L>
  struct TimeSeriesDatasetHeader {
    const DatasetHeader<L>& get_header() const {
      const auto& ds = static_cast<const Base&>(*this);
      const std::map<std::string, libtempo::DTS<F, L>>& map = *ds.dataset_shared_map;
      return map.begin()->second.header();
    }
  };
   */

  /// Interface for both the train and the test state
  template<typename Derived>
  struct IState {

    /// Action when creating a leaf
    virtual void on_leaf(const BCMVec& /* bcmvec */) = 0;

    /// Fork the state for sub branch/sub tree with index "bidx", leading to a "sub state"
    virtual Derived branch_fork(size_t /* bidx */) = 0;

    /// Merge-move "other" into "this".
    virtual void branch_merge(Derived&& /* other */) = 0;

    /// Clone at the forest level - clones must be fully independent as they can be used in parallel
    virtual std::unique_ptr<Derived> forest_fork(size_t /* tree_idx */) = 0;

    virtual ~IState() = default;
  };

  ///  Helper for state components.
  ///  We do not differentiate between branch and forest levels.
  ///  Do not use when dealing with "resources", e.g. a random number generator with internal state.
  template<typename DerivedComp>
  struct IStateComp {

    /// Action when creating a leaf
    virtual void on_leaf(const BCMVec& /* bcmvec */) = 0;

    /// Fork the state for sub branch/sub tree with index "bidx", leading to a "sub state"
    virtual DerivedComp fork(size_t /* bidx */) = 0;

    /// Merge "other" into "this".
    virtual void merge(const DerivedComp& /* other */) = 0;

    virtual ~IStateComp() = default;
  };

}
