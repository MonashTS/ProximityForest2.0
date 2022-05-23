#pragma once

#include <tempo/utils/utils.hpp>

namespace tempo {

  /* How we manage datasets
   * We do not have a "dataset" class; we rather have a set of classes
   *
   * DatasetHeader (with LabelEncoder)
   *  - This class contains general information about a dataset.
   *    In this case, the "dataset" represents all the data (e.g. both train and test splits) by contiguous indexes
   *    in [0, N[. Each exemplar is associated to an optional<Label>.
   *  - It contains a label encoder. A model trained with a given label encoder must be used with the same encoder
   *    (or a super set, even though the "new label" can't be predicted)
   *
   * DatasetTransform
   *  - By definition, a transform is an element obtained from another by transformation
   *    A dataset has "raw data". This data can be transformed, e.g. scaling, taking the derivative...
   *    There is no reason to treat the data and the transforms differently
   *
   *  - Transforms are named; we reserve the name "default" for the raw data.
   *
   *  - Transforms only contain the data; their relation to a given dataset header is established through
   *    a link to such a header (here, a std::shared_ptr<DatasetHeader>), and by matching indexes in [0, N[
   *
   * DatasetSplit (with IndexSet and ByClassMap)
   *  - The previous classes works on all the data from a dataset. However, we usually want to work with splits,
   *    such as train/evaluation/test splits.
   *
   *  - A DatasetSplit is made of a transform (which gives access to the header) and a subset of index in [0, n[.
   *  The subset [0, n[ map into the dataset header set of index  [0, N[
   *
   */


  class LabelEncoder {

    /// Set of labels for the dataset, with index encoding
    std::map<L, EL> _label_to_index;

    /// Reverse mapping index top label - Note: EL must be an integral type for indexing purposes
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

  public:

    LabelEncoder() = default;

    LabelEncoder(LabelEncoder const& other) = default;

    LabelEncoder(LabelEncoder&& other) = default;

    LabelEncoder& operator =(LabelEncoder&& other) = default;

    /// Create a new encoder given a set of label
    explicit LabelEncoder(std::set<L> const& labels) { update(labels); }

    /// Copy other into this, then add unknown label from 'labels'
    LabelEncoder(LabelEncoder other, std::set<L> const& labels) : LabelEncoder(std::move(other)) {
      update(labels);
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Labels to indexes encoding (reverse from index_to_label)
    inline std::map<L, EL> const& label_to_index() const { return _label_to_index; }

    /// Indexes to labels encoding (reverse from label_to_index)
    inline std::vector<L> const& index_to_label() const { return _index_to_label; }

    /// Encode a label
    inline EL encode(L const& l) const { return _label_to_index.at(l); }

    /// Decode a label
    inline L decode(EL el) const { return _index_to_label[el]; }
  };

  class DatasetHeader : tempo::utils::Uncopyable {

    /// Name of the dataset - usually the actual dataset name, but could be anything.
    std::string _name{"AnonDatasetHeader"};

    /// Smallest series length
    size_t _length_min{0};

    /// Longest series length
    size_t _length_max{0};

    /// Original dimensions of the dataset
    size_t _nb_dimensions{1};

    /// Mapping between exemplar index and label
    std::vector<std::optional<L>> _labels{};

    /// Series with missing data
    std::vector<size_t> _missing{};

    /// Label Encoder
    LabelEncoder _label_encoder;

  public:

    // --- --- --- --- --- ---
    // Constructors and assignment operator

    DatasetHeader() = default;

    /// Constructor, building the set of labels from the labels vector
    /// @param name Name of the dataset
    /// @param labels Labels per instance, identifier by position.
    ///               The dataset header represents the instances by their index in [0 .. labels.size()[
    /// @param encoder Existing encoder
    DatasetHeader(
      std::string name,
      size_t minlength,
      size_t maxlength,
      size_t dimensions,
      std::vector<std::optional<L>>&& labels,
      std::vector<size_t>&& instance_with_missing,
      LabelEncoder encoder={}
    ) :
      _name(std::move(name)),
      _length_min(minlength),
      _length_max(maxlength),
      _nb_dimensions(dimensions),
      _labels(std::move(labels)),
      _missing(std::move(instance_with_missing)) {
      // Build the set and the encoder
      std::set<L> labelset;
      for (std::optional<L> const& ol : _labels) {
        if (ol) { labelset.insert(ol.value()); }
      }
      // Build a new encoder
      _label_encoder = LabelEncoder(std::move(encoder), labelset);
    }

    DatasetHeader(DatasetHeader&& other) = default;

    DatasetHeader& operator =(DatasetHeader&& other) = default;

    // --- --- --- --- --- ---
    // Dataset properties

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

    /// Number of dimensions
    inline size_t nb_dimensions() const { return _nb_dimensions; }

    /// Index of instances with missing data
    inline const std::vector<size_t>& instances_with_missing() const { return _missing; }

    /// Check if any exemplar contains a missing value (encoded with "NaN")
    inline bool has_missing_value() const { return !(_missing.empty()); }

    // --- --- --- --- --- ---
    // Label access

    /// Access the original label of an instance
    inline std::optional<L> const& original_label(size_t idx) const { return _labels[idx]; }

    /// Get the encoded label of an instance
    inline std::optional<EL> label(size_t idx) const {
      std::optional<L> const& ol = original_label(idx);
      if (ol) { return {_label_encoder.encode(ol.value())}; }
      else { return {}; }
    }

    /// Encode a Label
    inline EL encode(L const& l) const { return _label_encoder.encode(l); }

    /// Decode an Encoded Label (EL)
    inline L decode(EL el) const { return _label_encoder.decode(el); }

    /// Get the number of classes
    inline size_t nbclasses() const { return _label_encoder.index_to_label().size(); }

    /// Direct access to the label encoder
    inline LabelEncoder const& label_encoder() const { return _label_encoder; }

  };


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  template<typename T>
  class DatasetTransform : tempo::utils::Uncopyable {

    std::shared_ptr<DatasetHeader> _header;
    std::string _name{"AnonDatasetTransform"};
    std::vector<T> _storage;

  public:

    // --- --- --- --- --- ---
    // Constructors and assignment operator

    DatasetTransform() = default;

    DatasetTransform(std::shared_ptr<DatasetHeader> header, std::string name, std::vector<T>&& data) :
      _header(std::move(header)), _name(std::move(name)), _storage(std::move(data)) {}

    DatasetTransform(DatasetTransform&& other) = default;

    DatasetTransform& operator =(DatasetTransform&& other) = default;


    // --- --- --- --- --- ---
    // Access

    T const& at(size_t idx) const { return _storage[idx]; }

    T const& operator [](size_t idx) const { return at(idx); }

    auto begin() const { return _storage.cbegin(); }

    auto end() const { return _storage.cend(); }

    size_t size() const { return _storage.size(); }

    /// Name separator
    inline static std::string sep = ";";

    /// Transform name
    std::string const& name() const { return _name; }

    /// Access to the header
    inline DatasetHeader const& header() const { return *_header; }

    /// Access to the header pointer
    inline std::shared_ptr<DatasetHeader> header_ptr() const { return _header; }

  };


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

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

    /// Create an index set of size N with [start, start+N[ (top excluded)
    explicit IndexSet(size_t start, size_t N) {
      vset = std::make_shared<std::vector<size_t>>(N);
      std::iota(vset->begin(), vset->end(), start);
    }

    /// Create an index set of size N [0, N[ (top excluded)
    explicit IndexSet(size_t N) : IndexSet(0, N) {}

    /// Set of index based on a collection.
    explicit IndexSet(std::vector<size_t>&& collection) {
      using std::begin, std::end;
      vset = std::make_shared<std::vector<size_t>>(std::move(collection));
    }

    /// Create a new IndexSet from another IndexSet and a selection of indexes
    /// @param other              Other IndexSet
    /// @param indexes_in_other   Iterable collection
    template<typename Collection>
    IndexSet(IndexSet const& other, Collection const& indexes_in_other) {
      using std::begin, std::end;
      // Test requested subset
      assert(indexes_in_other.size()>other.vset->size());
      std::vector<size_t> nv;
      nv.reserve(other.vset->size());
      for (auto i : indexes_in_other) { nv.push_back(other.vset->at(i)); }
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

  /// ByClassMap (BCM): a type gathering indexes in a dataset by encoded label
  class ByClassMap {
  public:
    /// Type of the map
    using BCM_t = std::map<EL, IndexSet>;

    /// Type of the map with a modifiable vector. Used in helper constructor.
    using BCMvec_t = std::map<EL, std::vector<size_t>>;

  private:
    BCM_t _bcm;
    size_t _size{0};
    std::map<EL, size_t> _map_index;
    std::set<EL> _classes;

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
      for (auto [l, v] : bcm) { _bcm.emplace(l, IndexSet(std::move(v))); }
      populate_indexes();
    }

    /// Build a BCM from a header and an index set
    /// If an exemplar has no label, it cannot be part of the BCM; its index is return in a vector
    inline static std::tuple<ByClassMap, std::vector<size_t>> make(DatasetHeader const& header, IndexSet const& is) {
      typename ByClassMap::BCMvec_t m;         // For index with label
      std::vector<size_t> v;                   // For index without label
      for (size_t idx : is) {
        const auto& olabel = header.label(idx);
        if (olabel.has_value()) { m[olabel.value()].push_back(idx); }   // vector: default constructed on 1st access
        else { v.push_back(idx); }                                      // No label here
      }
      return {ByClassMap(std::move(m)), std::move(v)};
    }



    // --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---  --- --- ---
    // Access

    /// Access the underlying map
    inline const BCM_t& operator *() const { return _bcm; }

    /// Bracket operator on the underlying map
    inline const auto& operator [](EL el) const { return _bcm.at(el); }

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
    inline const std::set<EL>& classes() const { return _classes; }

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

    /// Helper providing the by class (encoded label) cardinality in a Col vector
    /// Works for all labels present in the header, not just the one present in this BCM
    inline arma::Col<size_t> get_class_cardinalities(const DatasetHeader& header) {
      arma::Col<size_t> result(header.nbclasses(), arma::fill::zeros);
      for (const auto& [el, v] : *this) { result[el] = v.size(); }
      return result;
    }

    /// Encoded Label to index mapping: "compress" the encoded label into a [0, n[ range
    /// for the n labels present in this BCM
    inline const std::map<EL, size_t>& labels_to_index() const { return _map_index; }

    /// Convert this BCM into an IndexSet
    inline IndexSet to_IndexSet() const {
      // Reserve size for our vector, then, copy from BCM, sort, and build a new IndexSet
      std::vector<size_t> v;
      v.reserve(size());
      for (const auto& [_, is] : *this) { v.insert(v.end(), is.begin(), is.end()); }
      std::sort(v.begin(), v.end());
      return IndexSet(std::move(v));
    }
  };

  /// Manage a split: a combination of a dataset header with some stored data (transform) agreeing on their indexes,
  /// and a subset represented by IndexSet
  template<typename T>
  class DataSplit {

    std::string _name{"AnonDataSplit"};
    std::shared_ptr<DatasetTransform<T>> _transform{{}};
    IndexSet _index_set{};

    IndexSet const& index_set() const { return _index_set; }

  public:

    DataSplit() = default;

    DataSplit(DataSplit const& other) = default;

    DataSplit(DataSplit&& other) = default;

    /// Split with a name, store, and a subset of the transform
    DataSplit(std::string name, std::shared_ptr<DatasetTransform<T>> store, IndexSet is) :
      _name(std::move(name)),
      _transform(std::move(store)),
      _index_set(std::move(is)) {}

    /// Split with a name, store, and over the full transform
    DataSplit(std::string name, std::shared_ptr<DatasetTransform<T>> store) :
      _name(std::move(name)),
      _transform(std::move(store)),
      _index_set(0, _transform->size()) {}

    /// Subset of an existing split: is is indexing in other
    DataSplit(DataSplit const& other, std::string name, IndexSet is) :
      _name(std::move(name)),
      _transform(other._tranform),
      _index_set(other.index_set(), is) {}


    // --- --- --- --- --- ---
    // Access

    /// Access index within the split
    T const& operator [](size_t idx) const {
      assert(idx<size());
      size_t real_index = _index_set[idx];
      return _transform->at(real_index);
    }

    size_t size() const { return _index_set.size(); }

    // --- --- --- --- --- ---
    // Header & transform access

    /// Access to the transform
    inline DatasetTransform<T> const& transform() const { return *_transform; }

    /// Access to the transform pointer
    inline std::shared_ptr<DatasetTransform<T>> transform_ptr() const { return _transform; }

    /// Access to the header
    inline DatasetHeader const& header() const { return transform().header(); }

    /// Access to the header pointer
    inline std::shared_ptr<DatasetHeader> header_ptr() const { return transform().header_ptr(); }


    // --- --- --- --- --- ---
    // Label

    /// Access encoded label within the split
    std::optional<size_t> label(size_t idx) const {
      size_t real_index = _index_set[idx];
      return header().label(real_index);
    }

    /// Access original label within the split
    std::optional<L> original_label(size_t idx) const {
      size_t real_index = _index_set[idx];
      return header().original_label(real_index);
    }


    // --- --- --- --- --- ---
    // Name

    /// Name of the dataset, taken from the header
    std::string const& get_dataset_name() const { return _transform->header().name(); }

    /// Name of the associated data store
    std::string const& get_transform_name() const { return _transform->name(); }

    /// Name of this split
    std::string const& get_split_name() const { return _name; }

    /// Name separator
    inline static const std::string sep{"$"};

    /// Build the full name with the format "dataset<sep>transform<sep>split"
    /// Note: the transform has the format "a<DatasetTransform::sep>b..."
    std::string get_full_name() const {
      return get_dataset_name() + sep + get_transform_name() + sep + get_split_name();
    }

    // --- --- --- --- --- ---
    // BCM helper

    inline std::tuple<ByClassMap, std::vector<size_t>> get_BCM() const {
      return ByClassMap::make(header(), _index_set);
    }

  }; // End of DataSplit


} // end of namespace tempo