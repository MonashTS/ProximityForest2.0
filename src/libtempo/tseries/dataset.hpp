#pragma once

#include "tseries.hpp"

#include <libtempo/utils/utils.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cassert>
#include <memory>
#include <vector>
#include <set>

namespace libtempo {

  /// ByClassMap forward declaration
  class ByClassMap;

  /// Index Set:
  /// Manages a set of indexes, used to represent series from a Dataset (see below).
  class IndexSet {

  public:
    /// Subset by selection of indexes
    using VSet = std::shared_ptr<std::vector<size_t>>;

  private:
    VSet vset;

  public:

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Constructors

    /// Default constructor, building an empty set
    inline IndexSet() : IndexSet(0) {}

    /// Create an index set |0, top[ (top excluded)
    inline explicit IndexSet(size_t top) {
      vset = std::make_shared<std::vector<size_t>>(top);
      std::iota(vset->begin(), vset->end(), 0);
    }

    /// Set of index based on a collection. Collection must be ordered!
    inline explicit IndexSet(std::vector<size_t>&& collection) {
      using std::begin, std::end;
      assert(std::is_sorted(begin(collection), end(collection)));
      vset = std::make_shared<std::vector<size_t>>(std::move(collection));
    }

    /// Create a new IndexSet from another IndexSet and a selection of indexes
    /// @param other              Other IndexSet
    /// @param indexes_in_other   Vector of indexes i, indexing in other 0 <= i < other.size().
    ///                           Must be sorted
    inline IndexSet(const IndexSet& other, const std::vector<size_t>& indexes_in_other) {
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

    /// Index into the indexSet, returning a "real index" usable in a dataset.
    inline size_t get(size_t index) const { return vset->operator [](index); }

    /// Const Bracket operator aliasing 'get'
    inline size_t operator [](size_t index) const { return get(index); }

    /// Size of the set
    inline size_t size() const { return vset->size(); }

    /// Number of indexes contained
    inline bool empty() const { return vset->empty(); }

    /// Random Access Iterator begin
    inline auto begin() const { return vset->begin(); }

    /// Random Access Iterator end
    inline auto end() const { return vset->end(); }

    /// Access to the underlying vector
    inline const auto& vector() const { return *vset; }
  };

  /// ByClassMap (BCM): a tyype gathering indexes in a dataset by class
  /// Note: if an instance does not have a label (class), it cannot be part of a BCM.
  class ByClassMap {
  public:
    /// Type of the map
    using BCM_t = std::map<std::string, IndexSet>;

    /// Type of the map with a modifiable vector. Used in helper constructor.
    using BCMvec_t = std::map<std::string, std::vector<size_t>>;

  private:
    BCM_t _bcm;
    size_t _size{0};
    std::map<std::string, size_t> _map_index;
    std::set<std::string> _classes;

    /// Populate _indexes, _map_index and _classes
    inline void populate_indexes() {
      // Populate _map_index and _classes, compute the size
      size_t idx = 0;
      _size = 0;
      for (const auto& [l, is] : _bcm) {
        _classes.insert(l);
        _map_index[l] = idx;
        ++idx;
        _size += is.size();
      }
    }

  public:

    // --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---

    /// Default constructor
    inline ByClassMap() = default;

    /// Constructor taking ownership of a map<std::string, IndexSet>.
    /// The map is then used as provided.
    inline explicit ByClassMap(BCM_t&& bcm) : _bcm(std::move(bcm)) { populate_indexes(); }

    /// Constructor taking ownership of a map of <std::string, std::vector<size_t>>
    /// The vectors represent sets of index, and must be sorted (low to high)
    inline explicit ByClassMap(BCMvec_t&& bcm) {
      for (auto [l, v] : bcm) { _bcm.template emplace(l, IndexSet(std::move(v))); }
      populate_indexes();
    }

    // --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---
    // Access

    /// Access the underlying map
    inline const BCM_t& operator *() const { return _bcm; }

    /// Bracket operator on the underlying map
    inline const auto& operator [](const std::string& l) const { return _bcm.at(l); }

    /// Constant begin iterator on the underlying map - gives access on the label L and the associated IndexSet.
    inline auto begin() const { return _bcm.begin(); }

    /// Constant end iterator on the underlying map
    inline auto end() const { return _bcm.end(); }


    // --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---
    // Tooling

    /// Number of indexes contained
    inline size_t size() const { return _size; }

    /// Number of indexes contained
    inline bool empty() const { return _size==0; }

    /// Number of classes
    inline size_t nb_classes() const { return _classes.size(); }

    /// Set of classes
    inline const std::set<std::string>& classes() const { return _classes; }

    /// Randomly pick one index per class, returning a new ByClassMap with one item per class.
    template<typename PRNG>
    ByClassMap pick_one_by_class(PRNG& prng) const {
      BCM_t result;
      for (const auto& [label, is] : _bcm) {
        if (!is.empty()) {
          std::uniform_int_distribution<size_t> dist(0, is.size() - 1);
          result[label] = IndexSet(std::vector<size_t>{is[dist(prng)]});
        }
      }
      return ByClassMap(std::move(result));
    }

    /// Gini impurity of a BCM.
    /// Maximum purity is the minimum value of 0
    /// Minimum purity, i.e. maximum possible value, depends on the number of classes:
    ///   * 1-1/2=0.5 with 2 classes
    ///   * 1-1/3=0.6666..6 with 3 classes
    ///   * etc...
    inline double gini_impurity() const {
      assert(nb_classes()>0);
      // Ensure that we never encounter a "floating point near 0" issue.
      if (size()<=1) { return 0; }
      else {
        double total_size = size();
        double sum{0};
        for (const auto& [cl, val] : _bcm) {
          double p = val.size()/total_size;
          sum += p*p;
        }
        return 1 - sum;
      }
    }

    ///Label to index mapping
    inline const std::map<std::string, size_t>& labels_to_index() const { return _map_index; }

    /// Convert to an IndexSet
    inline IndexSet to_IndexSet() const {
      // Reserve size for our vector, then, copy from BCM, sort, and build a new IndexSet
      std::vector<size_t> v;
      v.reserve(size());
      for (const auto& [_, is] : *this) { v.insert(v.end(), is.begin(), is.end()); }
      std::sort(v.begin(), v.end());
      return IndexSet(std::move(v));
    }
  };

  /** The Dataset header is independent from the actual data -
   * it only records general information and the labels per instance.
   * An instance is represented by its index within [0, size[
   * where 'size' it the size of the dataset, i.e. the number of instances.
   * @tparam L  Type of the label
   */
  class DatasetHeader {
  public:
    using json = nlohmann::json;

  private:

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
    std::vector<std::optional<std::string>> _labels;

    /// Set of labels for the dataset, with index encoding
    std::map<std::string, size_t> _label_to_index;

    /// Reverse mapping index top label
    std::vector<std::string> _index_to_label;

  public:

    /** Constructor of a core dataset
     * @param dataset_name  A name use to identify this dataset
     * @param lmin          Minimum length of the series
     * @param lmax          Maximum length of the series
     * @param dimensions    Number of dimensions
     * @param labels        Label per instance, instances being indexed in [0, labels.size()[
     * @param instances_with_missing Indices of instances with missing data. Empty = no missing data.
     */
    inline DatasetHeader(
      std::string _name,
      size_t lmin,
      size_t lmax,
      size_t dimensions,
      std::vector<std::optional<std::string>>&& labels,
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
      std::set<std::string> lset;
      for (const auto& ol : _labels) { if (ol.has_value()) { lset.insert(ol.value()); }}
      // Set to map (Label, Index)
      size_t idx = 0;
      for (const auto& k : lset) {
        _label_to_index[k] = idx;
        _index_to_label.push_back(k);
        ++idx;
      }
    }

    /**   Given a set of index 'is', compute a tuple (BCM, vec) where:
     *  * 'BCM' gives a list of index from 'is' per label ("class")
     *  * 'vec' gives the index from 'is' associated to no label.
     *
     */
    inline std::tuple<ByClassMap, std::vector<size_t>> get_BCM(const IndexSet& is) const {
      typename ByClassMap::BCMvec_t m;         // For index with label
      std::vector<size_t> v;                   // For index without label
      for (size_t idx : is) {
        const auto& olabel = _labels[idx];
        if (olabel.has_value()) {
          m[olabel.value()].push_back(idx);
        }   // Label - vector default construction on 1st access
        else { v.push_back(idx); }                                      // No label here
      }
      return {ByClassMap(std::move(m)), std::move(v)};
    }

    /// Helper for the above, using all the index
    inline std::tuple<ByClassMap, std::vector<size_t>> get_BCM() const {
      return get_BCM(IndexSet(_size));
    }

    /// Build an indexset
    inline IndexSet index_set() const { return IndexSet(_size); }

    /// Base name of the dataset
    inline const std::string& name() const { return _name; }

    /// The size of the dataset, i.e. the number of exemplars
    inline size_t size() const { return _size; }

    /// The length of the shortest series in the dataset
    inline size_t length_min() const { return _length_min; }

    /// The length of the longest series in the dataset
    inline size_t length_max() const { return _length_max; }

    /// Check if all series have varying length (return true), or all have the same length (return false)
    inline bool variable_length() const { return _length_max!=_length_min; }

    inline size_t nb_dimensions() const { return _nb_dimensions; }

    /// Index of instances with missing data
    inline const std::vector<size_t>& instances_with_missing() const { return _instances_with_missing; }

    /// Check if any exemplar contains a missing value (encoded with "NaN")
    inline bool has_missing_value() const { return !(_instances_with_missing.empty()); }

    /// Label per instance. An instance may not have a label, hence we use optional
    inline const std::vector<std::optional<std::string>>& labels() const { return _labels; }

    /// Number of labels in the dataset
    inline size_t nb_labels() const { return _label_to_index.size(); }

    /// Labels to indexes encoding (reverse from index_to_label)
    inline const std::map<std::string, size_t>& label_to_index() const { return _label_to_index; }

    /// Indexes to labels encoding (reverse from label_to_index)
    inline const std::vector<std::string>& index_to_label() const { return _index_to_label; }

    /// Create a json representation of the header
    inline json to_json() const {
      json j;
      j["dataset_name"] = name();
      j["dataset_size"] = size();
      j["dataset_label"] = index_to_label();
      j["series_dimension"] = nb_dimensions();
      j["series_length_min"] = length_min();
      j["series_length_max"] = length_max();
      j["values_missing"] = has_missing_value();
      //
      return j;
    }

  };

  /** A dataset is a collection of data D, referring to a core dataset */
  template<typename D>
  class Dataset {

    using json = nlohmann::json;

    /// Reference to the core dataset. A datum indexed here must have a corresponding index in the core dataset.
    std::shared_ptr<DatasetHeader> _dataset_header;

    /// Identifier for this dataset. Can be used to record transformation, such as "derivative".
    std::string _identifier;

    /// Actual collection
    std::shared_ptr<std::vector<D>> _data;

    /// Optional parameter of the transform, as a JSONValue
    std::optional<json> _parameters;

  public:

    /// Generate an empty, non usable dataset
    Dataset() = default;

    /// Constructor: all data must be pre-constructed
    Dataset(
      std::shared_ptr<DatasetHeader> core,
      std::string id,
      std::vector<D>&& data,
      std::optional<json> parameters = {} // TODO
    )
      : _dataset_header(std::move(core)), _identifier(std::move(id)),
        _data(std::make_shared<std::vector<D>>(std::move(data))),
        _parameters(std::move(parameters)) {
      assert((bool)_dataset_header);
      assert(_data->size()==_dataset_header->size());
    }

    /// Constructor: using an existing dataset to get the dataset header.
    Dataset(
      const Dataset& other,
      std::string id,
      std::vector<D>&& data,
      std::optional<json> parameters = {} // TODO
    ) : _dataset_header(other._dataset_header),
        _identifier(std::move(id)),
        _data(std::make_shared<std::vector<D>>(std::move(data))),
        _parameters(std::move(parameters)) {
      assert((bool)_dataset_header);
      assert(_data->size()==_dataset_header->size());
    }

    /// Access to the vector of data
    const std::vector<D>& data() const { return *_data; }

    /// Access to the core dataset
    const DatasetHeader& header() const {
      assert((bool)_dataset_header);
      return *_dataset_header;
    }

    /// Access to the identifier
    const std::string& id() const { return _identifier; }

    /// Get the full name, made of the name of the ore dataset header, the identifier, and the parameters
    /// "name_in_header:id<JSONVALUE>"
    std::string name() const {
      assert((bool)_dataset_header);
      std::string result = _dataset_header->name() + ":" + _identifier;
      if (_parameters.has_value()) { result += _parameters.value().dump(); }
      return result;
    }

    /// Shorthand size
    inline size_t size() const { return _data->size(); }

    /// Shorthand []
    inline const D& operator [](size_t idx) const { return _data->operator [](idx); }

    /// Shorthand begin
    inline auto begin() const { return _data->begin(); }

    /// Shorthand end
    inline auto end() const { return _data->end(); }

  };

  /// Helper for Dataset of time series.
  template<Float F>
  using DTS = Dataset<TSeries<F>>;

  /// Helper for a DTS (Dataset of Time Series), computing statistics per dimension
  template<Float F>
  struct DTS_Stats {
    arma::Col<F> _min;
    arma::Col<F> _max;
    arma::Col<F> _mean;
    arma::Col<F> _stddev;

    DTS_Stats(const DTS<F>& dts, const IndexSet& is) {

      arma::running_stat_vec<arma::Col<F>> stat;
      for (const auto i : is) {
        const TSeries<F>& s = dts[i];
        const arma::Mat<F> mat = s.data();
        for (size_t c = 0; c<mat.n_cols; ++c) {
          stat(mat.col(c));
        }
      }

      _min = stat.min();
      _max = stat.max();
      _mean = stat.mean();
      _stddev = stat.stddev(0); // norm_type=0 performs normalisation using N-1 (N=number of samples)
    }

  };

  /// Helper for univariate DTS
  template<Float F>
  F stddev(const DTS<F>& dts, const IndexSet& is) {
    DTS_Stats<F> stat(dts, is);
    return stat._stddev[0];
  }

  /// Helper providing the by class cardinality in Col vector
  inline arma::Col<size_t> get_class_cardinalities(const DatasetHeader& header, const ByClassMap& bcm) {
    const auto& label_to_index = header.label_to_index();
    arma::Col<size_t> result(label_to_index.size(), arma::fill::zeros);
    for (const auto& [l, v] : bcm) { result[label_to_index.at(l)] = v.size(); }
    return result;
  }
}