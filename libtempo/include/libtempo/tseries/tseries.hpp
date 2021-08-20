#pragma once

#include <libtempo/utils/capsule.hpp>

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>
#include <cassert>

namespace libtempo::tseries {

  namespace { // Unnamed namespace: visibility local to the file - in header, do not put variable
    namespace lu = libtempo::utils;
  }

  template<typename LabelType, typename FloatType>
  class TSeries {
    static_assert(std::is_floating_point_v<FloatType>);
    static_assert(std::is_copy_constructible_v<LabelType>);

    /// Time series dimensions and length
    size_t _ndim{1};
    size_t _length{0};
    /// Size of the data: _ndim*_length
    size_t _size{0};
    /// Missing data in the time series? We use a floating point type, so should be represented by "nan"
    bool _missing{false};
    /// Optional label
    std::optional<LabelType> _olabel{};

    /** Actual data of the series:
     *  Two cases:
     *  - we do not own the data, we only have a pointer on it
     *  - we own the data: we put it in a capsule, and get the pointer out of it
     *  In both cases, we access the data through the 'data' pointer
     */
    lu::Capsule _c{}; // Remark: keep before *data - required for correct construction order.
    const FloatType* _data{nullptr};


    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Helpers
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    /** Common code for constructors - see them for params description */
    void init(
      size_t size, const FloatType* data,
      size_t ndim, std::optional<LabelType> olabel, std::optional<bool>omissing){
      // --- Checking
      if (ndim<1) { throw std::domain_error("Dimension can't be < 1"); }
      if (size%ndim!=0) { throw std::domain_error("Vector size is not a multiple of 'dim'"); }
      // --- Checking for missing data
      if(omissing.has_value()){
        _missing = omissing.value();
      } else {
        _missing = std::any_of(data, data+size, [](auto f){return std::isnan(f);});
      }
      // --- Complete init
      _ndim = ndim,
      _olabel = olabel,
      _length = size/_ndim;
      _size = size;
      _data = data;
    }


  public:

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Constructor
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Default constructor: create en univariate empty time series
    TSeries() = default;


    /** Create a TSeries from a raw pointer and associated information.
     *  Does not own the data. It is your responsibility to guarantee that the data live long enough.
     *  @param size The size of the @a data array. Equals the length of the series only if @a dim == 1
     *  @param data The series data, in "column major order", i.e. data must be grouped per points.
     *              For example, a multivariate series with dimensions A, B and C must be stored as
     *              [[A0, B0, C0], [A1, B1, C1] ... [An, Bn, Cn]]
     *  @param ndim The number of "dimension" (1 for univaraite, >1 for multivariate)
     *  @param olabel Optional label of the series
     *  @param omissing Optional information about missing data.
     *                  If none is provided (default), it is computed by checking for NaN
     */
    TSeries(size_t size, const FloatType* data, size_t ndim, std::optional<LabelType> olabel={}, std::optional<bool>omissing={}){
      init(size, data, ndim, olabel, omissing);
    }

    /** Create a TSeries from raw data and associated information
     *  Take ownership of the data
     *  @param data Vector of FloatType moved in the TSeries
     *  @param dim The number of "dimension" (1 for univariate, >1 for multivariate)
     *  @param olabel Optional label of the series
     *  @param omissing Optional information about missing data.
     *                  If none is provided (default), it is computed by checking for NaN
     */
    TSeries(std::vector<FloatType>&& data, size_t ndim, std::optional<LabelType> olabel={}, std::optional<bool>omissing={}) {
      size_t size = data.size();
      _c = lu::make_capsule(std::move(data));
      const auto* ptr = lu::capsule_ptr<std::vector<FloatType>>(_c);
      init(size, ptr, ndim, olabel, omissing);
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Accessors
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Access to the underlying data pointer
    [[nodiscard]] inline const FloatType* data(){return _data;}

    /// Access the number of dimensions
    [[nodiscard]] inline const size_t ndim(){return _ndim;}

    /// Access the length
    [[nodiscard]] inline const size_t length(){return _length;}

    /// Access the size of the data == ndim*length
    [[nodiscard]] inline const size_t size(){return _size;}

    /// Check if has missing values
    [[nodiscard]] inline const bool missing(){return _missing;}

    /// Get the label (perform a copy)
    [[nodiscard]] inline std::optional<LabelType> label(){return _olabel;}

    /// Access to the data with (timestamp, dimension)
    [[nodiscard]] inline FloatType operator()(size_t ts, size_t dim=0){
      assert(dim<_ndim);
      const size_t idx = ts*_ndim+dim;
      return _data[idx];
    }

    /// Pointer to the dimensions at a timestamp
    [[nodiscard]] inline const FloatType* operator[](size_t ts){
      const size_t idx = ts*_ndim;
      return _data+idx;
    };

  };

} // End of namespace libtempo::tseries
