#pragma once

#include "tseries.hpp"
#include <utility>
#include <tempo/utils/utils.hpp>

namespace tempo {

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
    IndexSet() : IndexSet(0) {}


    /// Create an index set of size N with |start, start+N[ (top excluded)
    explicit IndexSet(size_t start, size_t N) {
      vset = std::make_shared<std::vector<size_t>>(N);
      std::iota(vset->begin(), vset->end(), start);
    }

    /// Create an index set of size N |0, N[ (top excluded)
    explicit IndexSet(size_t N):IndexSet(0, N) {}

    /// Set of index based on a collection. Collection must be ordered!
    explicit IndexSet(std::vector<size_t>&& collection) {
      using std::begin, std::end;
      assert(std::is_sorted(begin(collection), end(collection)));
      vset = std::make_shared<std::vector<size_t>>(std::move(collection));
    }

    /// Create a new IndexSet from another IndexSet and a selection of indexes
    /// @param other              Other IndexSet
    /// @param indexes_in_other   Vector of indexes i, indexing in other 0 <= i < other.size().
    ///                           Must be sorted
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

  /// Label encoder
  class LabelEncoder {

    /// Set of labels for the dataset, with index encoding
    std::map<L, size_t> _label_to_index;

    /// Reverse mapping index top label
    std::vector<L> _index_to_label;

    /// Helper function: create the mapping structures with the content from the set _labels.
    /// Also use as an update function if _label has been *extended*.
    template<typename Collection>
    void update(Collection const& labels) {
      size_t idx = _index_to_label.size();
      for (auto const& k : labels) {
        if (!_label_to_index.contains(k)) {
          _label_to_index[k] = idx;
          _index_to_label.push_back(k);
          ++idx;
        }
      }
    }

    static std::set<L> to_set(std::vector<std::optional<L>> const& vec) {
      std::set<L> s;
      for (auto const& l : vec) { if (l) { s.insert(l.value()); }}
      return s;
    }

  public:

    LabelEncoder() = default;

    LabelEncoder(LabelEncoder const& other) = default;

    LabelEncoder(LabelEncoder&& other) = default;

    LabelEncoder& operator =(LabelEncoder&& other) = default;

    /// Create a new encoder given a set of label
    explicit LabelEncoder(std::set<L> const& labels) { update(labels); }

    /// Copy other into this, then add unknown label from 'labels'
    explicit LabelEncoder(LabelEncoder other, std::set<L> const& labels) : LabelEncoder(std::move(other)) {
      update(labels);
    }

    /// Helper with a vector of optional labels
    explicit LabelEncoder(std::vector<std::optional<L>> const& labels) :
      LabelEncoder(to_set(labels)) {}

    /// Helper with a vector of optional labels
    explicit LabelEncoder(LabelEncoder other, std::vector<std::optional<L>> const& labels) :
      LabelEncoder(std::move(other), to_set(labels)) {}

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Number of labels in the dataset
    inline size_t nb_labels() const { return _label_to_index.size(); }

    /// Labels to indexes encoding (reverse from index_to_label)
    inline const std::map<std::string, size_t>& label_to_index() const { return _label_to_index; }

    /// Indexes to labels encoding (reverse from label_to_index)
    inline const std::vector<std::string>& index_to_label() const { return _index_to_label; }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  };

  /** The Dataset header is independent from the actual data -
   * it only records general information and the labels per instance.
   * An instance is represented by its index within [0, size[
   * where 'size' it the size of the dataset, i.e. the number of instances.
   * @tparam L  Type of the label
   */
  class DatasetHeader {

    /// Identifier for the core dataset - usually the actual dataset name, but could be anything.
    std::string _name{"null"};

    /// Smallest series length
    size_t _length_min{0};

    /// Longest series length
    size_t _length_max{0};

    /// Original dimensions of the dataset
    size_t _nb_dimensions{1};

    /// Series with missing data
    std::vector<size_t> _missing{};

    /// Mapping between exemplar index and label
    std::vector<std::optional<L>> _labels{};

    ///
    LabelEncoder _label_encoder{};

  public:

    DatasetHeader() = default;

    /// Move constructor
    DatasetHeader(DatasetHeader&&) = default;

    /// Move assignment operator
    DatasetHeader& operator =(DatasetHeader&&) = default;

    /** Constructor of a core dataset
     * @param dataset_name  A name use to identify this dataset
     * @param lmin          Minimum length of the series
     * @param lmax          Maximum length of the series
     * @param dimensions    Number of dimensions
     * @param labels        Label per instance, instances being indexed in [0, labels.size()[
     * @param instances_with_missing Indices of instances with missing data. Empty = no missing data.
     * @param lencoder      Pre-existing label encoder
     */
    inline DatasetHeader(
      std::string _name,
      size_t lmin,
      size_t lmax,
      size_t dimensions,
      std::set<L> const& labelset,
      std::vector<std::optional<L>>&& labels,
      std::vector<size_t>&& instances_with_missing,
      LabelEncoder lencoder
    )
      : _name(std::move(_name)),
        _length_min(lmin),
        _length_max(lmax),
        _nb_dimensions(dimensions),
        _missing(std::move(instances_with_missing)),
        _labels(std::move(labels)),
        _label_encoder(std::move(lencoder), labelset) {}

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
      std::set<L> const& labelset,
      std::vector<std::optional<L>>&& labels,
      std::vector<size_t>&& instances_with_missing
    ) :
      DatasetHeader(std::move(_name), lmin, lmax, dimensions, labelset, std::move(labels), std::move(instances_with_missing),
                    LabelEncoder()) {}

    /** Merge two existing dataset header into a new one.
     *  The only constraint is a match on the number of dimensions (throw std::logic if they do not match).
     *  The new dataset D is a concatenation of the data in 'a' follow 'b',
     *  i.e. the data from a are at indexes [0..a-1], and the data from b are at [a..b-1].
     *  A new label encoder is created from the all the collected labels.
     *  Label can be renamed on the fly with the transformation function.
     */
    inline DatasetHeader(
      DatasetHeader a,
      DatasetHeader b,
      std::string name,
      std::optional<std::function<std::optional<L>(std::optional<L> const&)>> rename_a = {},
      std::optional<std::function<std::optional<L>(std::optional<L> const&)>> rename_b = {}
    ) : _name(std::move(name)) {
      if (a.nb_dimensions()!=b.nb_dimensions()) {
        throw std::logic_error("DatasetHeader merging: no common dimension");
      }
      _length_min = std::min(a._length_min, b._length_min);
      _length_max = std::max(a._length_max, b._length_max);
      _nb_dimensions = a._nb_dimensions;
      // Labels
      _labels.reserve(a.size() + b.size());
      // A
      std::copy(a._labels.begin(), a._labels.end(), std::back_inserter(_labels));
      if (rename_a) {
        std::transform(_labels.begin(), _labels.end(), _labels.begin(), rename_a.value());
      }
      // B
      std::copy(b._labels.begin(), b._labels.end(), std::back_inserter(_labels));
      if (rename_b) {
        std::transform(_labels.begin() + a.size(), _labels.end(), _labels.begin() + a.size(), rename_b.value());
      }
      // Instance with missing
      _missing.reserve(a.instances_with_missing().size() + b.instances_with_missing().size());
      std::copy(a.instances_with_missing().begin(), a.instances_with_missing().end(), std::back_inserter(_missing));
      std::copy(b.instances_with_missing().begin(), b.instances_with_missing().end(), std::back_inserter(_missing));
      // New label encoder made from both dataset header
      _label_encoder = LabelEncoder(_labels);
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
      return get_BCM(IndexSet(size()));
    }

    /// Build an indexset
    inline IndexSet index_set() const { return IndexSet(size()); }

    /// Base name of the dataset
    inline const std::string& name() const { return _name; }

    /// The size of the dataset, i.e. the number of exemplars
    inline size_t size() const { return _labels.size(); }

    /// The length of the shortest series in the dataset
    inline size_t length_min() const { return _length_min; }

    /// The length of the longest series in the dataset
    inline size_t length_max() const { return _length_max; }

    /// Check if all series have varying length (return true), or all have the same length (return false)
    inline bool variable_length() const { return _length_max!=_length_min; }

    inline size_t nb_dimensions() const { return _nb_dimensions; }

    /// Index of instances with missing data
    inline const std::vector<size_t>& instances_with_missing() const { return _missing; }

    /// Check if any exemplar contains a missing value (encoded with "NaN")
    inline bool has_missing_value() const { return !(_missing.empty()); }

    /// Label per instance. An instance may not have a label, hence we use optional
    inline const std::vector<std::optional<L>>& labels() const { return _labels; }

    /// Label for a given instance
    inline std::optional<L> label(size_t i) const { return _labels[i]; }

    /// Index in the label encoder for a given instance.
    inline std::optional<size_t> label_index(size_t i) const {
      auto const& l = _labels[i];
      if (l) { return {_label_encoder.label_to_index().at(l.value())}; }
      else { return {}; }
    }

    /// Access the label encoder
    inline LabelEncoder const& label_encoder() const { return _label_encoder; }

    /// Create a json representation of the header
    inline Json::Value to_json() const {
      Json::Value j;
      j["dataset_name"] = name();
      j["dataset_size"] = size();
      j["dataset_label"] = utils::to_json(_label_encoder.index_to_label());
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

    /// Reference to the core dataset. A datum indexed here must have a corresponding index in the core dataset.
    std::shared_ptr<DatasetHeader> _dataset_header;

    /// Actual collection
    std::shared_ptr<std::vector<D>> _data;

    /// Transform name of this dataset, e.g. "derivative".
    std::string _transform_name;

    /// Optional parameter of the transform, as a JSONValue
    std::optional<Json::Value> _parameters;

  public:

    /// Generate an empty, non usable dataset
    Dataset() = default;

    /// Constructor: all data must be pre-constructed
    Dataset(
      std::shared_ptr<DatasetHeader> header,
      std::string transform_name,
      std::vector<D>&& data,
      std::optional<Json::Value> parameters = {}
    ) : _dataset_header(std::move(header)),
        _data(std::make_shared<std::vector<D>>(std::move(data))),
        _transform_name(std::move(transform_name)),
        _parameters(std::move(parameters)) {
      assert((bool)_dataset_header);
      assert(_data->size()==_dataset_header->size());
    }

    /// Constructor: using an existing dataset to get the dataset header.
    Dataset(
      const Dataset& other,
      std::string transform_name,
      std::vector<D>&& data,
      std::optional<Json::Value> parameters = {}
    ) : _dataset_header(other._dataset_header),
        _data(std::make_shared<std::vector<D>>(std::move(data))),
        _transform_name(std::move(transform_name)),
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
    const std::string& transform_name() const { return _transform_name; }

    /// Get the full name, made of the name from the dataset header and the transform name
    /// E.g. "name_in_header:derivative1"
    std::string fullname() const {
      assert((bool)_dataset_header);
      std::string result = _dataset_header->name() + ":" + transform_name();
      if (_parameters.has_value()) { result += _parameters.value().toStyledString(); }
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
  using DTS = Dataset<TSeries>;

  /// Map of named DTS
  using DTSMap = std::map<std::string, DTS>;

  /// Helper for a DTS (Dataset of Time Series), computing statistics per dimension
  struct DTS_Stats {
    arma::Col<F> _min;
    arma::Col<F> _max;
    arma::Col<F> _mean;
    arma::Col<F> _stddev;

    DTS_Stats(const DTS& dts, const IndexSet& is) {

      arma::running_stat_vec<arma::Col<F>> stat;
      for (const auto i : is) {
        const TSeries& s = dts[i];
        const arma::Mat<F>& mat = s.data();
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
  inline F stddev(const DTS& dts, const IndexSet& is) {
    DTS_Stats stat(dts, is);
    return stat._stddev[0];
  }

  /// Helper providing the by class cardinality in Col vector
  inline arma::Col<size_t> get_class_cardinalities(const DatasetHeader& header, const ByClassMap& bcm) {
    const auto& label_to_index = header.label_encoder().label_to_index();
    arma::Col<size_t> result(label_to_index.size(), arma::fill::zeros);
    for (const auto& [l, v] : bcm) { result[label_to_index.at(l)] = v.size(); }
    return result;
  }
}