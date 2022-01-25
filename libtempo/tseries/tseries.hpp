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