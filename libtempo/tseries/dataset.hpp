#pragma once

#include <libtempo/tseries/tseries.hpp>
#include <libtempo/concepts.hpp>
#include <libtempo/utils/jsonvalue.hpp>

#include <cassert>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <ranges>
#include <set>
#include <string>
#include <map>
#include <utility>
#include <vector>
#include <iostream>

namespace libtempo {

  /** ByClassMap forward declaration */
  template<Label L>
  class ByClassMap;

  /** Index Set:
   * Manages a set of indexes, used to represent series from a Dataset (see below). */
  class IndexSet {

  public:
    /** Subset by selection of indexes */
    using VSet = std::shared_ptr<std::vector<size_t>>;

  private:
    VSet vset;

  public:

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Constructors

    /** Default constructor, building an empty set */
    IndexSet() : IndexSet(0) {}

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

    /** Build from a ByClassMap */
    template<Label L>
    explicit IndexSet(const ByClassMap<L>& bcm) {
      // Reserve size for our vector, then, copy from BCM, sort, and build a new IndexSet
      std::vector<size_t> v;
      v.reserve(bcm.size());
      for (const auto&[_, is] : bcm) { v.insert(v.end(), is.begin(), is.end()); }
      std::sort(v.begin(), v.end());
      vset = std::make_shared<std::vector<size_t>>(std::move(v));
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
      for (auto i : indexes_in_other) { nv.push_back(ov->at(i)); }
      vset = std::make_shared<std::vector<size_t>>(std::move(nv));
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Accesses

    /** Index into the indexSet, returning a "real index" usable in a dataset. */
    [[nodiscard]]
    size_t get(size_t index) const { return vset->operator [](index); }

    /** Const Bracket operator aliasing 'get' */
    [[nodiscard]]
    size_t operator [](size_t index) const { return get(index); }

    /** Size of the set */
    [[nodiscard]]
    size_t size() const { return vset->size(); }

    /** Number of indexes contained */
    [[nodiscard]]
    bool empty() const { return vset->empty(); }

    /** Random Access Iterator begin */
    [[nodiscard]]
    auto begin() const { return vset->begin(); }

    /** Random Access Iterator end */
    [[nodiscard]]
    auto end() const { return vset->end(); }

  };

  /** ByClassMap (BCM): a tyype gathering indexes in a dataset by class
   *  Note: if an instance does not have a label (class), it cannot be part of a BCM.*/
  template<Label L>
  class ByClassMap {
  public:
    /// Type of the map
    using BCM_t = std::map<L, IndexSet>;

    /// Type of the map with a modifiable vector. Used in helper constructor.
    using BCMvec_t = std::map<L, std::vector<size_t>>;

  private:
    BCM_t _bcm;
    size_t _size{0};
    std::map<L, size_t> _map_index;
    std::set<L> _classes;

    /// Populate _indexes, _map_index and _classes
    void populate_indexes() {
      // Populate _map_index and _classes, compute the size
      size_t idx = 0;
      _size = 0;
      for (const auto&[l, is] : _bcm) {
        _classes.insert(l);
        _map_index[l] = idx;
        ++idx;
        _size += is.size();
      }
    }

  public:

    // --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---

    /// Default constructor
    ByClassMap() = default;

    explicit ByClassMap(BCM_t&& bcm) : _bcm(std::move(bcm)) {
      populate_indexes();
    }

    explicit ByClassMap(BCMvec_t&& bcm) {
      for (auto[l, v] : bcm) { _bcm.template emplace(l, IndexSet(std::move(v))); }
      populate_indexes();
    }

    // --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---
    // Access

    /// Access the underlying map
    [[nodiscard]]
    const BCM_t& operator *() const { return _bcm; }

    /// Bracket operator on the underlying map
    [[nodiscard]]
    const auto& operator [](L l) const { return _bcm.at(l); }

    /// Constant begin iterator on the underlying map - gives access on the label L and the associated IndexSet.
    [[nodiscard]]
    auto begin() const { return _bcm.begin(); }

    /// Constant end iterator on the underlying map
    [[nodiscard]]
    auto end() const { return _bcm.end(); }


    // --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---
    // Tooling

    /** Number of indexes contained */
    [[nodiscard]]
    size_t size() const { return _size; }

    /** Number of indexes contained */
    [[nodiscard]]
    bool empty() const { return _size==0; }

    /** Number of classes */
    [[nodiscard]]
    size_t nb_classes() const { return _classes.size(); }

    /** Set of classes */
    [[nodiscard]]
    const std::set<L>& classes() const { return _classes; }

    /** Randomly pick one index per class, returning a new ByClassMap with one item per class. */
    template<typename PRNG>
    [[nodiscard]]
    ByClassMap<L> pick_one_by_class(PRNG& prng) const {
      BCM_t result;
      for (const auto&[label, is] : _bcm) {
        if (is.size()>0) {
          std::uniform_int_distribution<size_t> dist(0, is.size() - 1);
          result[label] = IndexSet(std::vector<size_t>{is[dist(prng)]});
        }
      }
      return ByClassMap(std::move(result));
    }

    /** Gini impurity of a BCM.
     *  Maximum purity is the minimum value of 0
     *  Minimum purity, i.e. maximum possible value, depends on the number of classes:
     *    * 1-1/2=0.5 with 2 classes
     *    * 1-1/3=0.6666..6 with 3 classes
     *    * etc...
     */
    [[nodiscard]]
    double gini_impurity() const {
      assert(!empty());
      // Ensure that we never encounter a "floating point near 0" issue.
      if (size()==1) { return 0; }
      else {
        double total_size = size();
        double sum{0};
        for (const auto&[cl, val] : _bcm) {
          double p = val.size()/total_size;
          sum += p*p;
        }
        return 1 - sum;
      }
    }

    /** Label to index mapping */
    [[nodiscard]]
    const std::map<L, size_t>& labels_to_index() const { return _map_index; }

  };

  /** The Dataset header is independent from the actual data -
   * it only records general information and the labels per instance.
   * An instance is represented by its index within [0, size[
   * where 'size' it the size of the dataset, i.e. the number of instances.
   * @tparam L  Type of the label
   */
  template<Label L>
  class DatasetHeader {

    /// Identifier for the core dataset - usually the actual dataset name.
    std::string _name;

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

    /// Set of labels for the dataset, with index encoding
    std::map<L, size_t> _label_to_index;

    /// Reverse mapping index top label
    std::map<size_t, L> _index_to_label;

  public:

    /** Constructor of a core dataset
     * @param dataset_name  A name use to identify this dataset
     * @param lmin          Minimum length of the series
     * @param lmax          Maximum length of the series
     * @param dimensions    Number of dimensions
     * @param labels        Label per instance, instances being indexed in [0, labels.size()[
     * @param instances_with_missing Indices of instances with missing data. Empty = no missing data.
     */
    DatasetHeader(
      std::string _name,
      size_t lmin,
      size_t lmax,
      size_t dimensions,
      std::vector<std::optional<L>>&& labels,
      std::vector<size_t>&& instances_with_missing
    )
      : _name(std::move(_name)),
        _size(labels.size()),
        _length_min(lmin),
        _length_max(lmax),
        _nb_dimensions(dimensions),
        _instances_with_missing(std::move(instances_with_missing)),
        _labels(std::move(labels)) {
      // Set of labels from input vector: guarantee uniqueness
      std::set<L> lset;
      for (const auto& ol : _labels) { if (ol.has_value()) { lset.insert(ol.value()); }}
      // Set to map (Label, Index)
      size_t idx = 0;
      for (auto k : lset) {
        _label_to_index[k] = idx;
        _index_to_label[idx] = k;
        ++idx;
      }
    }

    /**   Given a set of index 'is', compute a tuple (BCM, vec) where:
     *  * 'BCM' gives a list of index from 'is' per label ("class")
     *  * 'vec' gives the index from 'is' associated to no label.
     *
     */
    [[nodiscard]] inline std::tuple<ByClassMap<L>, std::vector<size_t>> get_BCM(const IndexSet& is) const {
      typename ByClassMap<L>::BCMvec_t m;         // For index with label
      std::vector<size_t> v;                      // For index without label
      for (size_t idx : is) {
        const auto& olabel = _labels[idx];
        if (olabel.has_value()) { m[olabel.value()].push_back(idx); }   // Label - vector default construction on 1st access
        else { v.push_back(idx); }                                      // No label here
      }
      return {ByClassMap<L>(std::move(m)), std::move(v)};
    }

    /// Helper for the above, using all the index
    [[nodiscard]]
    inline std::tuple<ByClassMap<L>, std::vector<size_t>> get_BCM() const {
      return get_BCM(IndexSet(_size));
    }

    /// Base name of the dataset
    [[nodiscard]]
    inline const std::string& name() const { return _name; }

    /// The size of the dataset, i.e. the number of exemplars
    [[nodiscard]]
    inline size_t size() const { return _size; }

    /// The length of the shortest series in the dataset
    [[nodiscard]]
    inline size_t length_min() const { return _length_min; }

    /// The length of the longest series in the dataset
    [[nodiscard]]
    inline size_t length_max() const { return _length_max; }

    /// Check if all series have varying length (return true), or all have the same length (return false)
    [[nodiscard]]
    inline bool variable_length() const { return _length_max!=_length_min; }

    [[nodiscard]]
    inline size_t nb_dimensions() const { return _nb_dimensions; }

    /// Index of instances with missing data
    [[nodiscard]]
    inline const std::vector<size_t>& instances_with_missing() const { return _instances_with_missing; }

    /// Check if any exemplar contains a missing value (encoded with "NaN")
    [[nodiscard]]
    inline bool has_missing_value() const { return !(_instances_with_missing.empty()); }

    /// Label per instance. An instance may not have a label, hence we use optional
    [[nodiscard]]
    inline const std::vector<std::optional<L>>& labels() const { return _labels; }

    /// Number of labels in the dataset
    [[nodiscard]]
    inline size_t nb_labels() const { return _label_to_index.size(); }

    /// Labels to indexes encoding (reverse from index_to_label)
    [[nodiscard]]
    inline const std::map<L, size_t>& label_to_index() const { return _label_to_index; }

    /// Indexes to labels encoding (reverse from label_to_index)
    [[nodiscard]]
    inline const std::map<size_t, L>& index_to_label() const { return _index_to_label; }

  };

  /** A dataset is a collection of data D, referring to a core dataset */
  template<Label L, typename D>
  class Dataset {

    /// Reference to the core dataset. A datum indexed here must have a corresponding index in the core dataset.
    std::shared_ptr<DatasetHeader<L>> _dataset_header;

    /// Identifier for this dataset. Can be used to record transformation, such as "derivative".
    std::string _identifier;

    /// Actual collection
    std::shared_ptr<std::vector<D>> _data;

    /// Optional parameter of the transform, as a JSONValue
    std::optional<tempo::json::JSONValue> _parameters;

  public:

    /// Constructor: all data must be pre-constructed
    Dataset(
      std::shared_ptr<DatasetHeader<L>> core,
      std::string id,
      std::vector<D>&& data,
      std::optional<tempo::json::JSONValue> parameters = {} // TODO
    )
      : _dataset_header(core), _identifier(std::move(id)), _data(std::make_shared<std::vector<D>>(std::move(data))),
        _parameters(std::move(parameters)) {
      assert((bool)_dataset_header);
      assert(_data->size()==_dataset_header->size());
    }

    /// Constructor: using an existing dataset to get the dataset header.
    Dataset(
      const Dataset& other,
      std::string id,
      std::vector<D>&& data,
      std::optional<tempo::json::JSONValue> parameters = {} // TODO
    ) : _dataset_header(other._dataset_header),
        _identifier(std::move(id)),
        _data(std::make_shared<std::vector<D>>(std::move(data))),
        _parameters(std::move(parameters)) {
      assert((bool)_dataset_header);
      assert(_data->size()==_dataset_header->size());
    }

    /// Access to the vector of data
    [[nodiscard]] const std::vector<D>& data() const { return *_data; }

    /// Access to the core dataset
    [[nodiscard]] const DatasetHeader<L>& header() const {
      assert((bool)_dataset_header);
      return *_dataset_header;
    }

    /// Access to the identifier
    [[nodiscard]] const std::string& id() const { return _identifier; }

    /// Get the full name, made of the name of the ore dataset header, the identifier, and the parameters
    /// "name_in_header:id<JSONVALUE>"
    [[nodiscard]] std::string name() const {
      assert((bool)_dataset_header);
      std::string result = _dataset_header->name() + ":" + _identifier;
      if (_parameters.has_value()) { result += to_string_inline(_parameters.value()); }
      return result;
    }

    /// Shorthand size
    [[nodiscard]] inline size_t size() const { return _data->size(); }

    /// Shorthand []
    [[nodiscard]] inline const D& operator [](size_t idx) const { return _data->operator [](idx); }

    /// Shorthand begin
    [[nodiscard]] inline auto begin() const { return _data->begin(); }

    /// Shorthand end
    [[nodiscard]] inline auto end() const { return _data->end(); }

  };

  /// Helper for Dataset of time series.
  template<Float F, Label L>
  using DTS = Dataset<L, TSeries<F, L>>;

}