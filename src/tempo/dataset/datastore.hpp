#pragma once

#include <tempo/utils/utils.hpp>

namespace tempo {

  class LabelEncoder : tempo::utils::Uncopyable {

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

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Labels to indexes encoding (reverse from index_to_label)
    inline std::map<L, size_t> const& label_to_index() const { return _label_to_index; }

    /// Indexes to labels encoding (reverse from label_to_index)
    inline std::vector<L> const& index_to_label() const { return _index_to_label; }

    /// Encode a label
    inline size_t encode(L const& l) const { return _label_to_index.at(l); }

    /// Decode a label
    inline L decode(size_t i) const { return _index_to_label[i]; }
  };

  template<typename T>
  class TransformStore : tempo::utils::Uncopyable {

    /// Name of the transformation stored here like "derivative1"
    std::string _name{"AnonTransformStore"};
    std::vector<T> _storage;

  public:

    // --- --- --- --- --- ---
    // Constructors and assignment operator

    TransformStore() = default;

    TransformStore(std::string name, std::vector<T>&& data) :
      _name(std::move(name)), _storage(std::move(data)) {}

    TransformStore(TransformStore&& other) = default;

    TransformStore& operator =(TransformStore&& other) = default;


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
    DatasetHeader(
      std::string name,
      size_t minlength,
      size_t maxlength,
      size_t dimensions,
      std::vector<std::optional<L>>&& labels,
      std::vector<size_t>&& instance_with_missing
    ) :
      _name(std::move(name)),
      _length_min(minlength),
      _length_max(maxlength),
      _nb_dimensions(dimensions),
      _labels(std::move(labels)),
      _missing(std::move(instance_with_missing)) {
      // Build the set and the encoder
      std::set<L> labelset;
      for (std::optional<L> const& ol : labels) {
        if (ol) { labelset.insert(ol.value()); }
      }
      _label_encoder = LabelEncoder(labelset);
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
    inline std::optional<size_t> label(size_t idx) const {
      std::optional<L> const& ol = original_label(idx);
      if (ol) { return {_label_encoder.encode(ol.value())}; }
      else { return {}; }
    }

    /// Encode a label
    inline size_t encode(L const& l) const { return _label_encoder.encode(l); }

    /// Decode a label
    inline L decode(size_t i) const { return _label_encoder.decode(i); }

  };

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
    explicit IndexSet(size_t N) : IndexSet(0, N) {}

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

  /// Manage a split: a combination of a dataset header with some stored data (transform) agreeing on their indexes,
  /// and a subset represented by IndexSet
  template<typename T>
  class Split {

    std::string _name{"AnonSplit"};
    std::shared_ptr<DatasetHeader> _dataset_header{{}};
    std::shared_ptr<TransformStore<T>> _data_store{{}};
    IndexSet _index_set{};

  public:

    Split() = default;

    Split(Split const& other) = default;

    Split(Split&& other) = default;

    Split(std::string name,
          std::shared_ptr<DatasetHeader> header,
          std::shared_ptr<TransformStore<T>> store,
          IndexSet is) :
      _name(std::move(name)),
      _dataset_header(std::move(header)),
      _data_store(std::move(store)),
      _index_set(std::move(is)) {}


    // --- --- --- --- --- ---
    // Access

    /// Access index within the split
    T const& operator [](size_t idx) const {
      assert(idx<size());
      size_t real_index = _index_set[idx];
      return _data_store->at(real_index);
    }

    size_t size() const { return _index_set.size(); }


    // --- --- --- --- --- ---
    // Label

    /// Access encoded label within the split
    std::optional<size_t> label(size_t idx) const {
      size_t real_index = _index_set[idx];
      return _dataset_header->label(real_index);
    }

    /// Access original label within the split
    std::optional<L> original_label(size_t idx) const {
      size_t real_index = _index_set[idx];
      return _dataset_header->original_label(real_index);
    }


    // --- --- --- --- --- ---
    // Name

    /// Name of the dataset, taken from the header
    std::string const& get_dataset_name() const { return _dataset_header->name(); }

    /// Name of the associated data store
    std::string const& get_transform_name() const { return _data_store->name(); }

    /// Name of this split
    std::string const& get_split_name() const { return _name; }

    /// Name separator
    inline static const std::string sep{"$"};

    /// Build the full name with the format "dataset<sep>transform<sep>split"
    /// Note: the transform has the format "a<TransformStore::sep>b..."
    std::string get_full_name() const {
      return get_dataset_name() + sep + get_transform_name() + sep + get_split_name();
    }

  };

} // end of namespace tempo