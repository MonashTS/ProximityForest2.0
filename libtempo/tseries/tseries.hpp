#pragma once

#include <libtempo/utils/capsule.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>
#include <cassert>

namespace libtempo {

  namespace { // Unnamed namespace: visibility local to the file (when in header, do not declare variable here!)
    namespace lu = libtempo::utils;
  }

  template<typename FloatType, typename LabelType>
  class TSeries {
    static_assert(std::is_floating_point_v<FloatType>);
    static_assert(std::is_copy_constructible_v<LabelType>);

    using EArrayCM = Eigen::Array<FloatType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using EArrayRM = Eigen::Array<FloatType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // --- --- --- Shape

    /// Size: actual number of items (= _nbvar * _length)
    size_t _size{0};
    /// Number of variables
    size_t _nvar{1};
    /// Number of observation per variables - length of the series
    size_t _length{0};

    // --- --- --- Meta

    /// Missing data in the time series? We use a floating point type, so should be represented by "nan"
    bool _missing{false};

    /// Optional label
    std::optional<LabelType> _olabel{};

    // --- --- --- We use the Eigen library to represent the data.

    // Data are always stored in a "Dense" way (i.e. not sparse).
    // We have two scenarios:
    //    * We own the data: we move the data in a capsule and build an Eigen Map on top of it
    //    * We do not own the data: we receive a raw pointer, on top of which we build an Eigen Map
    //    In both cases, we manipulate Eigen Map
    //
    // Another consideration is the storage order. By default, C and C++ are "row major",
    // i.e. a matrix M is "flatten" row by row.
    // The other alternative, "column major", is to "flatten" the matrix colum by column.
    // This is of interest in the case of multivariate series
    // (there is no difference for univariate accesses).
    // Multivariate series are stored row by row (one row for the observations of one variable),
    // meaning that the data of one variable is stored contiguously in a "row major" order.
    // However, we usually need to get the data per timestamp.
    // To efficiently access per timestamp, we need to store the series in "column major" order
    // so that observation per time stamp (and not per variable) are now stored contiguously.
    //
    // Eigen allows to rewrite the data in another order.

    // Capsule: used when we own the data

    // Row major
    lu::Capsule _c_RM{};
    FloatType const* _p_RM{nullptr};
    Eigen::Map<const EArrayRM> _m_RM{nullptr, 0, 0};

    // Column major
    lu::Capsule _c_CM{};
    FloatType const* _p_CM{nullptr};
    Eigen::Map<const EArrayCM> _m_CM{nullptr, 0, 0};

  private:

    TSeries(
      // Row major data
      lu::Capsule&& crm,
      FloatType const* prm,
      Eigen::Map<const EArrayRM>&& mrm,
      // Column major data
      lu::Capsule&& ccm,
      FloatType const* pcm,
      Eigen::Map<const EArrayCM>&& mcm,
      // Dimensions
      size_t nbvar,
      size_t length,
      // Other
      std::optional<LabelType> olabel,
      bool has_missing
    )
      :
      _size(mrm.size()),
      _nvar(nbvar),
      _length(length),
      _missing(has_missing),
      _olabel(olabel),
      // RM
      _c_RM(std::move(crm)),
      _p_RM(std::move(prm)),
      _m_RM(std::move(mrm)),
      // CM
      _c_CM(std::move(ccm)),
      _p_CM(std::move(pcm)),
      _m_CM(std::move(mcm)) { }


  public:

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Construction
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /** Default constructor: create en univariate empty time series */
    TSeries() = default;

    static TSeries mk_rowmajor(
      std::vector<FloatType>&& v,
      size_t nbvar,
      std::optional<LabelType> olabel,
      std::optional<bool> omissing) {

      using namespace std;

      // --- Checking

      size_t vsize = v.size();
      if (nbvar<1) { throw domain_error("Number of variable can't be < 1"); }
      if (vsize%nbvar!=0) { throw domain_error("Vector size is not a multiple of 'nbvar'"); }
      size_t length = vsize/nbvar;

      // --- Build data

      // Row major- - take ownership of the incoming vector
      auto crm = lu::make_capsule<vector<FloatType>>(move(v));
      FloatType const* prm = lu::get_capsule_ptr<vector<FloatType>>(crm)->data();
      Eigen::Map<const EArrayRM> mrm(prm, nbvar, length);

      // Column major - allocate our own
      EArrayCM acm = mrm;   // Create column major array
      auto ccm = lu::make_capsule<EArrayCM>(move(acm));
      FloatType const* pcm = lu::get_capsule_ptr<EArrayCM>(ccm)->data();
      Eigen::Map<const EArrayCM> mcm(pcm, nbvar, length);

      // Check missing values
      bool has_missing = false;
      if (omissing.has_value()) { has_missing = omissing.value(); }
      else { has_missing = mrm.hasNaN(); }

      // Build the TSeries
      return TSeries(
        // RM
        move(crm), prm, move(mrm),
        // CM
        move(ccm), pcm, move(mcm),
        // Dimension
        nbvar, length,
        // Other
        olabel, has_missing
      );
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Basic access
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Access number of variable
    [[nodiscard]] inline size_t nvar() const { return _nvar; }

    /// Access the length
    [[nodiscard]] inline size_t length() const { return _length; }

    /// Access the size of the data == ndim*length
    [[nodiscard]] inline size_t size() const { return _size; }

    /// Check if has missing values
    [[nodiscard]] inline bool missing() const { return _missing; }

    /// Get the label (perform a copy)
    [[nodiscard]] inline std::optional<LabelType> label() const { return _olabel; }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Data access
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// RM - Row Major access to the raw data
    [[nodiscard]] inline FloatType const* rm_data() const { return _p_RM; }

    /// RM - Row Major access to the Eigen Map
    [[nodiscard]] inline Eigen::Map<const EArrayRM> rm_emap() const { return _m_RM; }

    /// CM - Column Major access to the raw data
    [[nodiscard]] inline FloatType const* cm_data() const { return _p_CM; }

    /// CM - Column Major access to the Eigen Map
    [[nodiscard]] inline Eigen::Map<const EArrayCM> cm_emap() const { return _m_CM; }



  }; // End of class TSeries

} // End of namespace libtempo







//
//
//
//    /** Create a TSeries from a raw pointer and associated information.
//     *  Does not own the data. It is your responsibility to guarantee that the data live long enough.
//     *  @param size The size of the data array. Equals the length of the series only if dim == 1
//     *  @param data The series data, in "column major order", i.e. data must be grouped per points.
//     *              For example, a multivariate series with dimensions A, B and C must be stored as
//     *              [[A0, B0, C0], [A1, B1, C1] ... [An, Bn, Cn]]
//     *  @param ndim The number of "dimension" (1 for univariate, >1 for multivariate)
//     *  @param olabel Optional label of the series
//     *  @param omissing Optional information about missing data.
//     *                  If none is provided (default), it is computed by checking for NaN
//     */
//    TSeries(
//      size_t size,
//      FloatType const* data,
//      size_t ndim,
//      std::optional<LabelType> olabel = {},
//      std::optional<bool> omissing = {}) {
//      init(size, data, ndim, olabel, omissing);
//    }
//
//
//    /// ROW MAJOR order (C order)
//    lu::Capsule _c_RM{};
//    /// Pointer on RM
//    FloatType const* _data_rm{nullptr};
//
//    /// COLUMN MAJOR order (Fortran order)
//    lu::Capsule _c_CM{};
//    /// Pointer on CM
//    FloatType const* _data_cm{nullptr};
//
//    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
//    // Helpers
//    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
//
//    /// Row Major (C style) to Column Major (Fortran style)
//    /// For example, with nvar = 3, length = 4, we have a transformation
//    /// from [[A0 A1 A2 A3][B0 B1 B2 B3][C0 C1 C2 C3]] to [[A0 B0 C0][A1 B1 C1][A2 B2 C2][A3 B3 C3]]
//    void mk_cm_from_rm() {
//      // Ensure we haven't any data yet
//      assert(_data_cm==nullptr);
//      assert(!_c_CM->has_value());
//      // Allocate a new vector, move it in a capsule, and take the pointer on it's data
//      std::vector<FloatType> cmdata(_size);
//      _c_CM = lu::make_capsule(std::move(cmdata));
//      _data_cm = lu::get_capsule_ptr<std::vector<FloatType>>(_c_CM)->data();
//      // Transpose RM to CM
//      for (size_t li = 0; li<_nvar; ++li) {
//        for (size_t co = 0; co<_length; ++co) {
//          _data_cm[co*_nvar+li] = _data_rm[li*_length+co];
//        }
//      }
//    }
//
//    /// Opposite as above
//    void mk_rm_from_cm() {
//      // Ensure we haven't any data yet
//      assert(_data_rm==nullptr);
//      assert(!_c_RM->has_value());
//      // Allocate a new vector, move it in a capsule, and take the pointer on it's data
//      std::vector<FloatType> rmdata(_size);
//      _c_RM = lu::make_capsule(std::move(rmdata));
//      _data_rm = lu::get_capsule_ptr<std::vector<FloatType>>(_c_RM)->data();
//      // Transpose CM to RM
//      for (size_t li = 0; li<_length; ++li) {
//        for (size_t co = 0; co<_nvar; ++co) {
//          _data_rm[co*_length+li] = _data_cm[li*_nvar+co];
//        }
//      }
//    }
//
//  public:
//
//    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
//    // Time series information
//    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
//
//    /// Access the number of variables
//    [[nodiscard]] inline size_t nvar() const { return _nvar; }
//
//    /// Access the length
//    [[nodiscard]] inline size_t length() const { return _length; }
//
//    /// Access the size of the data == ndim*length
//    [[nodiscard]] inline size_t size() const { return _size; }
//
//    /// Check if has missing values
//    [[nodiscard]] inline bool missing() const { return _missing; }
//
//    /// Get the label (perform a copy)
//    [[nodiscard]] inline std::optional<LabelType> label() const { return _olabel; }
//
//    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
//    // Row Major Time series data access
//    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
//
//    /// Access to the underlying data pointer - Row Major
//    [[nodiscard]] inline FloatType const* data_rm() const { return _data_rm; }
//
//    /// Row Major access to the data at "variable" (access to all timestamps of a variable)
//    [[nodiscard]] inline FloatType const* at_rm(size_t var) const {
//      assert(var>=0 && var<_nvar);
//      const size_t idx = var*_length;
//      return &_data_rm[idx];
//    }
//
//    /// Row Major access to the data with (variable, timestamp)
//    [[nodiscard]] inline FloatType at_rm(size_t var, size_t ts) const {
//      assert(var>=0 && var<_nvar);
//      assert(ts>=0 && ts<_length);
//      const size_t idx = var*_length+ts;
//      return _data_rm[idx];
//    }
//
//    /// Return a row major accessor with (variable, timestamp).
//    [[nodiscard]] inline auto get_rm_accessor() const {
//      return [=, capture = _c_RM](size_t var, size_t ts) {
//        const size_t idx = var*_length+ts;
//        return _data_rm[idx];
//      };
//    }
//
//    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
//    // Column Major Time series data access
//    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
//
//    /// Access to the underlying data pointer - Column Major
//    [[nodiscard]] inline FloatType const* data_cm() const { return _data_cm; }
//
//    /// Column Major access to the data at "timestamp" (access to all the variable at timestamps)
//    [[nodiscard]] inline FloatType const* at_cm(size_t ts) const {
//      assert(ts>=0 && ts<_length);
//      const size_t idx = ts*_nvar;
//      return &_data_cm[idx];
//    }
//
//    /// Column Major access to the data with (variable, timestamp)
//    [[nodiscard]] inline FloatType at_cm(size_t var, size_t ts) const {
//      assert(var>=0 && var<_nvar);
//      assert(ts>=0 && ts<_length);
//      const size_t idx = ts*_nvar+var;
//      return _data_cm[idx];
//    }
//
//    /// Return a row major accessor with (variable, timestamp).
//    [[nodiscard]] inline auto get_cm_accessor() const {
//      return [=, capture = _c_CM](size_t var, size_t ts) {
//        assert(var>=0 && var<_nvar);
//        assert(ts>=0 && ts<_length);
//        const size_t idx = ts*_nvar+var;
//        return _data_cm[idx];
//      };
//    }



///** Common code for constructors
// * @param size Size of the data, in number of data point (i.e. number of dimensions * length)
// * @param data Pointer on the data
// * @param ndim Number of dimensions
// * @param olabel Optional label
// * @param omissing Optional information about missing data.
// *                 If not provided, all the data will be tested with "isnan"
// */
//void init(
//  size_t size,
//  const FloatType *data,
//  size_t ndim,
//  std::optional<LabelType> olabel,
//  std::optional<bool> omissing
//) {
//  // --- Checking (dimension, length) & size info
//  if (ndim < 1) { throw std::domain_error("Dimension can't be < 1"); }
//  if (size % ndim != 0) { throw std::domain_error("Vector size is not a multiple of 'dim'"); }
//  // --- Checking for missing data
//  if (omissing.has_value()) { _missing = omissing.value(); }
//  else { _missing = std::any_of(data, data + size, [](auto f) { return std::isnan(f); }); }
//  // --- Complete init
//  _nvar = ndim;
//  _olabel = olabel;
//  _length = size / _ndim;
//  _size = size;
//}

//
//    /** Create a TSeries from raw data and associated information
//     *  Take ownership of the data
//     *  @param data Vector of FloatType moved in the TSeries
//     *  @param dim The number of "dimension" (1 for univariate, >1 for multivariate)
//     *  @param olabel Optional label of the series
//     *  @param omissing Optional information about missing data.
//     *                  If none is provided (default), it is computed by checking for NaN
//     */
//    TSeries(std::vector<FloatType>&& data, size_t ndim, std::optional<LabelType> olabel = {},
//      std::optional<bool> omissing = {}) {
//      size_t size = data.size();
//      _c = lu::make_capsule(std::move(data));
//      auto const* ptr = lu::get_capsule_ptr<std::vector<FloatType>>(_c);
//      init(size, ptr, ndim, olabel, omissing);
//    }
//
//    /// Check that two series data are equals. The label is excluded from the check.
//    friend bool operator==(const TSeries& c1, const TSeries& c2);
//  };

/*
  /// Check that two series data are equals. The label is excluded from the check.
  /// Note: having "nan" in the same place in both series is considered as equal
  /// even if floating point arithmetic usually consider that nan != nan.
  template<typename FloatType, typename LabelType>
  bool operator==(const TSeries<FloatType, LabelType>& t1, const TSeries<FloatType, LabelType>& t2) {
    // Check dimensions and some info about the values
    bool is_equal = t1.size()==t2.size() && t1.nvar()==t2.nvar() && t1.missing()==t2.missing();
    // Check the values using the row major representation (the column major would also work)
    auto const* p1 = t1.data_rm();
    auto const* p2 = t2.data_rm();
    for (size_t i = 0; is_equal && i<t1.size(); ++i) {
      const auto i1 = p1[i];
      const auto i2 = p2[i];
      is_equal = i1==i2 || (std::isnan(i1) && std::isnan(i2));
    }
    return is_equal;
  }

  */
