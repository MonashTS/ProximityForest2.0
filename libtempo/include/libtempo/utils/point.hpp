#pragma once

namespace libtempo::utils {

  /** Structure representing a series point at a given timestamp.
   * @tparam FloatType The host type of the series
   * Note: the dimension of the series is not recorded
   * The users of the point should store this data and act accordingly
   */
  template <typename FloatType>
  struct point {
    const FloatType* data{nullptr};       /// Pointer to the start of the data
    size_t ts{0};                         /// Timestamp
  };


}