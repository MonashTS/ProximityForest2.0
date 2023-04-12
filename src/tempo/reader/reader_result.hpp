#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/utils/label_encoder.hpp>

namespace tempo::reader {

  template<std::floating_point F>
  struct ReaderData {

    /// Storing the data of a series as a vector - gives contiguous block of memory
    /// Note: The length of a series is the length of the vector devided by the number of dimension.
    /// Note: Assume row-major
    using SData = std::vector<F>;

    /// The collection of series from the dataset; a series is represented by its index.
    std::vector<SData> series;

    /// Optionally, if series have labels, we have a vector of encoded label (matching series by index)
    /// See the LabelEncoder 'encoder'
    std::optional<std::vector<std::string>> labels;

    /// Label Encoder
    LabelEncoder encoder;

    /// Indexes of the series containing NaN
    std::set<size_t> series_with_nan;

    /// Smallest length read
    size_t length_min{};

    /// Largest length read
    size_t length_max{};

    /// Number of dimensions
    size_t nb_dimensions{};

    // --- --- ---

    // Do not copy me!
    ReaderData(ReaderData const& other) = delete;
    ReaderData& operator =(ReaderData const& other) = delete;

    // Move me instead!
    ReaderData(ReaderData&&) noexcept = default;
    ReaderData& operator =(ReaderData&&) noexcept = default;

    ReaderData() = default;
  };

  /// Result type for the readers; any error shall be written in the variant's string
  template<std::floating_point F>
  using Result = std::variant<std::string, ReaderData<F>>;

} // End of namespace tempo