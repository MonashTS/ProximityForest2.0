#pragma once

#include <libtempo/concepts.hpp>
#include <libtempo/utils/jsonvalue.hpp>

#include <cassert>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace libtempo {

  /** ByClassMap (BCM): a tyype gathering indexes in a dataset by class
   *  Note: if an instance does not have a label (class), it cannot be part of a BCM.*/
  template<Label L>
  using ByClassMap = std::unordered_map<L, std::vector<size_t>>;

  /** Given a BCM, randomly pick one index per class
   *  The random number generator is provided as an argument,
   *  and used with modification (it updates its internal state: cannot be const; do not copy it!).
   *  If a BCM has a mapping between a label and an empty list of index, it will be removed from the result.
   */
  template<Label L, typename PRNG>
  [[nodiscard]] inline ByClassMap<L> pick_one_by_class(const ByClassMap<L>& bcm, PRNG& prng) {
    ByClassMap<L> result;
    for (const auto&[c, v]: bcm) {
      if (v._size()>0) {
        std::uniform_int_distribution<size_t> dist(0, v._size()-1);
        result[c].push_back(v[dist(prng)]);
      }
    }
    return result;
  }

  /** Compute the size of a BCM in number of indexes it contains */
  template<typename L>
  [[nodiscard]] inline size_t size(const ByClassMap<L>& bcm) {
    size_t s = 0;
    for (const auto&[_, v]: bcm) { s += v._size(); }
    return s;
  }

  /** Gini impurity of a BCM.
   *  Maximum purity is the minimum value of 0 (only one class)
   *  Minimum purity depends on the number of classes:
   *    * 1-1/2=0.5 with 2 classes
   *    * 1-1/3=0.6666..6 with 3 classes
   *    * etc...
   */
  template<typename L>
  [[nodiscard]] inline double gini_impurity(const ByClassMap<L>& bcm) {
    assert(!bcm.empty());
    // Ensure that we never encounter a "floating point near 0" issue.
    if (bcm.size()==1) { return 0; }
    else {
      double total_size = size(bcm);
      double sum{0};
      for (const auto&[cl, val]: bcm) {
        double p = val._size()/total_size;
        sum += p*p;
      }
      return 1-sum;
    }
  }

  /** Index Set */
  class IndexSet {

  public:

    /** Subset by selection of indexes */
    using VSet = std::shared_ptr<std::vector<size_t>>;

  private:
    VSet vset;

  public:

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Constructors

    /** Create an index set |0, top[ (top excluded) */
    explicit IndexSet(size_t top) {
      vset = std::make_shared<std::vector<size_t>>(top);
      std::iota(vset->begin(), vset->end(), 0);
    }

    /** Set of index based on a collection. Collection must be ordered! */
    explicit IndexSet(std::vector<size_t>&& collection) {
      using std::begin, std::end;
      assert(std::is_sorted(begin(collection), end(collection)));
      vset = std::make_shared<std::vector<size_t>>(std::move(collection));
    }

    /** Create an IndexSet over a ByClassMap */
    template<Label L>
    explicit IndexSet(const ByClassMap<L>& bcm) {
      using std::begin, std::end;
      std::shared_ptr<std::vector<size_t>> idx = std::make_shared<std::vector<size_t>>();
      for (const auto&[c, v]: bcm) { (*idx).insert(end(*idx), begin(v), end(v)); }
      std::sort(idx->begin(), idx->end());
      vset = std::move(idx);
    }

    /** Create a new dataset from another dataset and a selection of indexes
    * @param other             Other dataset
    * @param indexes_in_other  Vector of indexes i, indexing in other 0 <= i < other.size(). Must be sorted
    */
    IndexSet(const IndexSet& other, const std::vector<size_t>& indexes_in_other) {
      using std::begin, std::end;
      // Test requested subset
      assert(indexes_in_other.size()>other.vset->size());
      assert(std::is_sorted(begin(indexes_in_other), end(indexes_in_other)));
      const VSet& ov = other.vset;
      std::vector<size_t> nv;
      nv.reserve(ov->size());
      for (auto i: indexes_in_other) { nv.push_back(ov->at(i)); }
      vset = std::make_shared<std::vector<size_t>>(std::move(nv));
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Accesses

    /** Index into the indexSet, returning a "real index" usable in a dataset. */
    [[nodiscard]] inline size_t get(size_t index) const { return vset->operator[](index); }

    /** Const Bracket operator aliasing 'get' */
    [[nodiscard]] inline size_t operator[](size_t index) const { return get(index); }

    /** Size of the set */
    [[nodiscard]] inline size_t size() const { return vset->size(); }

    /** Random Access Iterator begin */
    [[nodiscard]] inline auto begin() const { return vset->begin(); }

    /** Random Access Iterator end */
    [[nodiscard]] inline auto end() const { return vset->end(); }
  };

  /** The Core Dataset is independent from the actual data -
   * it only records general information and the labels per instance.
   * An instance is represented by its index within [0, size[
   * where 'size' it the size of the dataset, i.e. the number of instances.
   * @tparam L  Type of the label
   */

  template<Label L>
  class CoreDataset {

    /// Identifier for the core dataset - usually the actual dataset name.
    std::string _dataset_name;

    /// Size of the dataset in number of instances. Instances are indexed in [0, size[
    size_t _size;

    /// Smallest series length
    size_t _length_min;

    /// Longest series length
    size_t _length_max;

    /// Original dimensions of the dataset
    size_t _nb_dimensions;

    /// Series with missing data
    std::vector<size_t> _instances_with_missing;

    /// Mapping between index and label
    std::vector<std::optional<L>> _labels;

    /// Set of labels for the dataset
    std::set<L> _label_set;

  public:

    /** Constructor of a core dataset
     * @param dataset_name  A name use to identify this dataset
     * @param lmin          Minimum length of the series
     * @param lmax          Maximum length of the series
     * @param dimensions    Number of dimensions
     * @param labels        Label per instance, instances being indexed in [0, labels.size()[
     * @param instances_with_missing Indices of instances with missing data. Empty = no missing data.
     */
    CoreDataset(
      std::string dataset_name,
      size_t lmin,
      size_t lmax,
      size_t dimensions,
      std::vector<std::optional<L>>&& labels,
      std::vector<size_t>&& instances_with_missing
    )
      :_dataset_name(std::move(dataset_name)),
       _size(labels.size()),
       _length_min(lmin),
       _length_max(lmax),
       _nb_dimensions(dimensions),
       _instances_with_missing(std::move(instances_with_missing)),
       _labels(std::move(labels)) {
      /// Construct set of labels from the input vector
      std::set<L> lset;
      for (const auto& ol: labels) { if (ol.has_value()) { lset.insert(ol.value()); }}
      _label_set = lset;
    }

    /// Get indexes by label, return a tuple (BCM, vec),
    /// where 'BCM' gives a list of indices per label ("class"), and 'vec' gives the indices associated to no label.
    [[nodiscard]] inline std::tuple<ByClassMap<L>, std::vector<size_t>> get_BCM(const IndexSet& is) const {
      ByClassMap<L> m;        // For index with label
      std::vector<size_t> v;  // For index without label
      auto it = is.begin();
      auto end = is.end();
      while (it!=end) {
        const auto idx = *it;
        const auto& olabel = _labels[idx];
        if (olabel.has_value()) {
          m[olabel.value()].push_back(idx); // Note: default construction of the vector of first access
        } else {
          v.push_back(idx);                 // No label here
        }
        ++it;
      }
      return {m, v};
    }

    [[nodiscard]] inline const std::string& dataset_name() const { return _dataset_name; }

    [[nodiscard]] inline size_t size() const { return _size; }

    [[nodiscard]] inline size_t length_min() const { return _length_min; }

    [[nodiscard]] inline size_t length_max() const { return _length_max; }

    [[nodiscard]] inline size_t nb_dimensions() const { return _nb_dimensions; }

    [[nodiscard]] inline const std::vector<size_t>& instances_with_missing() const {
      return _instances_with_missing;
    }

    [[nodiscard]] inline const std::vector<std::optional<L>>& labels() const { return _labels; }

    [[nodiscard]] inline const std::set<L>& label_set() const { return _label_set; }

  };

  /** A dataset is a collection of data D, referring to a core dataset */
  template<Label L, typename D>
  class Dataset {

    /// Reference to the core dataset. A datum indexed here must have a corresponding index in the core dataset.
    std::shared_ptr<CoreDataset<L>> _core_dataset;

    /// Identifier for this dataset. Can be used to record transformation, such as "derivative".
    std::string _identifier;

    /// Actual collection
    std::vector<D> _data;

    /// Optional parameter of the transform, as a JSONValue
    std::optional<tempo::json::JSONValue> _parameters;

  public:

    /// Constructor: all data must be pre-constructed
    Dataset(
      std::shared_ptr<CoreDataset<L>> core,
      std::string id,
      std::vector<D>&& data,
      std::optional<tempo::json::JSONValue> parameters = {}
    )
      :_core_dataset(core), _identifier(std::move(id)), _data(std::move(data)), _parameters(std::move(parameters)) {
      assert(_data.size()==_core_dataset->size());
    }

    /// Constructor: using an existing dataset to get the core dataset.
    Dataset(
      const Dataset& other,
      std::string id,
      std::vector<D>&& data,
      std::optional<tempo::json::JSONValue> parameters = {}
    )
      :_core_dataset(other._core_dataset), _identifier(std::move(id)), _data(std::move(data)), _parameters(std::move(parameters)) {
      assert(_data.size()==_core_dataset->size());
    }

    /// Create json info


    /// Access to the vector of data
    [[nodiscard]] const std::vector<D>& data() const { return _data; }

    /// Access to the core dataset
    [[nodiscard]] const CoreDataset<L>& core() const { return *_core_dataset; }

    /// Access to the identifier
    [[nodiscard]] const std::string& id() const { return _identifier; }

    /// Get the full name, made of the name of the core dataset, the identifier, and the parameters
    /// "core:id<JSONVALUE>"
    [[nodiscard]] std::string name() const {
      std::string result = _core_dataset->dataset_name()+":"+_identifier;
      if(_parameters.has_value()){
        result+=to_string_inline(_parameters.value());
      }
      return result;
    }

    /// Shorthand size
    [[nodiscard]] inline size_t size() const { return _data.size(); }

    /// Shorthand []
    [[nodiscard]] inline const D& operator[](size_t idx) const { return _data[idx]; }

    /// Shorthand begin
    [[nodiscard]] inline auto begin() const { return _data.begin(); }

    /// Shorthand end
    [[nodiscard]] inline auto end() const { return _data.end(); }

    /// Shorthand missing value
    [[nodiscard]] inline bool has_missing_value() const { return !_core_dataset->instances_with_missing().empty(); }


  };

  template<Float F, Label L>
  using DTS = Dataset<L, TSeries<F, L>>;


}