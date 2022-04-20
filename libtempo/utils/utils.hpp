#pragma once

#include <stdexcept>
#include <limits>
#include <cassert>
#include <cmath>

#include "partasks.hpp"
#include "progressmonitor.hpp"
#include "stats.hpp"
#include "timing.hpp"

namespace libtempo::utils {

  /** Pick a random item from a subscriptable type, from [0] to [size-1]*/
  template<typename PRNG>
  [[nodiscard]] inline
  const auto& pick_one(const Subscriptable auto& collection, size_t size, PRNG& prng){
    if(size==1){
      return collection[0];
    } else if(size>1){
      auto distribution = std::uniform_int_distribution<size_t>(0, size - 1);
      return collection[distribution(prng)];
    } else {
      throw std::invalid_argument("Picking from an empty collection");
    }
  }

  /** Pick a random item from a vector. */
  template<typename T, typename PRNG>
  [[nodiscard]] const auto& pick_one(const std::vector<T>& v, PRNG& prng) {
    return pick_one(v, v.size(), prng);
  }


  // --- --- --- --- --- ---
  // --- Constants
  // --- --- --- --- --- ---

  /// Constant to be use when no window is required
  constexpr size_t NO_WINDOW{std::numeric_limits<size_t>::max()};

  /// Positive infinity for float types
  template<typename FloatType>
  constexpr FloatType PINF{std::numeric_limits<FloatType>::infinity()};

  /// Negative infinity for float types
  template<typename FloatType>
  constexpr FloatType NINF{-PINF<FloatType>};

  /// Not A Number
  template<typename FloatType>
  constexpr FloatType QNAN{std::numeric_limits<FloatType>::quiet_NaN()};

  /// Lower Bound inital value, use to deal with numerical instability
  template<typename FloatType>
  FloatType INITLB{-pow(FloatType(10), -(std::numeric_limits<FloatType>::digits10 - 1))};



  // --- --- --- --- --- ---
  // --- Tooling
  // --- --- --- --- --- ---

  /// Minimum of 3 values using std::min<T>
  template<typename T>
  [[nodiscard]] inline T min(T a, T b, T c) { return std::min<T>(a, std::min<T>(b, c)); }

  /// Maximum of 3 values using std::min<T>
  template<typename T>
  [[nodiscard]] inline T max(T a, T b, T c) { return std::max<T>(a, std::max<T>(b, c)); }



  // --- --- --- --- --- ---
  // --- Should not happen
  // --- --- --- --- --- ---

  /// Throw an exception "should not happen". Used as default case in switches.
  [[noreturn]] void inline should_not_happen() { throw std::logic_error("Should not happen"); }



  // --- --- --- --- --- ---
  // --- Unsigned tooling
  // --- --- --- --- --- ---


  /** Unsigned arithmetic:
   * Given an 'index' and a 'window', get the start index corresponding to std::max(0, index-window) */
  [[nodiscard]] inline size_t cap_start_index_to_window(size_t index, size_t window) {
    if (index>window) { return index - window; } else { return 0; }
  }

  /** Unsigned arithmetic:
   * Given an 'index', a 'window' and an 'end', get the stop index corresponding to std::min(end, index+window+1).
   * The expression index+window+1 is illegal for any index>0 as window could be MAX-1
   * */
  [[nodiscard]] inline size_t
  cap_stop_index_to_window_or_end(size_t index, size_t window, size_t end) {
    // end-window is valid when window<end
    if (window<end&&index + 1<end - window) { return index + window + 1; } else { return end; }
  }

  /** Absolute value for any comparable and subtractive type, without overflowing risk for unsigned types.
   *  Also work for signed type. */
  template<typename T>
  [[nodiscard]] inline T absdiff(T a, T b) { return (a>b) ? a - b : b - a; }

  /** From unsigned to signed for integral types*/
  template<typename UIType>
  [[nodiscard]] inline typename std::make_signed_t<UIType> to_signed(UIType ui) {
    static_assert(std::is_unsigned_v<UIType>, "Template parameter must be an unsigned type");
    using SIType = std::make_signed_t<UIType>;
    if (ui>(UIType)(std::numeric_limits<SIType>::max())) {
      throw std::overflow_error("Cannot store unsigned type in signed type.");
    }
    return (SIType)ui;
  }


  // --- --- --- --- --- ---
  // --- Initialisation tool
  // --- --- --- --- --- ---

  namespace initBlock_detail {
    struct tag {};

    template<class F>
    decltype(auto) operator +(tag, F&& f) {
      return std::forward<F>(f)();
    }
  }

#define initBlock initBlock_detail::tag{} + [&]() -> decltype(auto)

#define initBlockStatic initBlock_detail::tag{} + []() -> decltype(auto)

} // end of namespace libtempo::utils
