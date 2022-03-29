#pragma once

#include <libtempo/utils/capsule.hpp>
#include <libtempo/concepts.hpp>

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

  template<Float F, Label L>
  class TSeries {
    using EArrayCM = Eigen::Array<F, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using EArrayRM = Eigen::Array<F, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

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
    std::optional<L> _olabel{};

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
    F const* _p_RM{nullptr};
    Eigen::Map<const EArrayRM> _m_RM{nullptr, 0, 0};

    // Column major
    lu::Capsule _c_CM{};
    F const* _p_CM{nullptr};
    Eigen::Map<const EArrayCM> _m_CM{nullptr, 0, 0};

  private:

    TSeries(
      // Row major data
      lu::Capsule&& crm,
      F const* prm,
      Eigen::Map<const EArrayRM>&& mrm,
      // Column major data
      lu::Capsule&& ccm,
      F const* pcm,
      Eigen::Map<const EArrayCM>&& mcm,
      // Dimensions
      size_t nbvar,
      size_t length,
      // Other
      std::optional<L> olabel,
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

    /** Default move constructor */
    TSeries(TSeries&&)  noexcept = default;

    /** Disable copy (we have to move instead: do not duplicate data) */
    TSeries(const TSeries&) = delete;

    [[nodiscard]] static TSeries mk_rowmajor(
      std::vector<F>&& v,
      size_t nbvar,
      std::optional<L> olabel,
      std::optional<bool> omissing) {

      using namespace std;

      // --- Checking

      size_t vsize = v.size();
      if (nbvar<1) { throw domain_error("Number of variable can't be < 1"); }
      if (vsize%nbvar!=0) { throw domain_error("Vector size is not a multiple of 'nbvar'"); }
      size_t length = vsize/nbvar;

      // --- Build data

      // Row major- - take ownership of the incoming vector
      auto crm = lu::make_capsule<vector<F>>(move(v));
      F const* prm = lu::get_capsule_ptr<vector<F>>(crm)->data();
      Eigen::Map<const EArrayRM> mrm(prm, nbvar, length);

      // Column major - allocate our own
      EArrayCM acm = mrm;   // Create column major array
      auto ccm = lu::make_capsule<EArrayCM>(move(acm));
      F const* pcm = lu::get_capsule_ptr<EArrayCM>(ccm)->data();
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

    /** Copy data from other, except for the actual data. Allow to easily do transforms. No checking done. */
    [[nodiscard]] static TSeries mk_rowmajor(
      const TSeries& other,
      std::vector<F>&& v
    ) {
      return mk_rowmajor(std::move(v), other.nvar(), other.label(), {other.missing()});
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
    [[nodiscard]] inline std::optional<L> label() const { return _olabel; }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Data access
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Row Major access over all the point AND dimension -- shorthand for access using rm_data()
    [[nodiscard]] inline F operator[](size_t idx) const { return _p_RM[idx]; }

    /// RM - Row Major access to the raw data
    [[nodiscard]] inline F const* rm_data() const { return _p_RM; }

    /// RM - Row Major access to the Eigen Map
    [[nodiscard]] inline Eigen::Map<const EArrayRM> rm_emap() const { return _m_RM; }

    /// CM - Column Major access to the raw data
    [[nodiscard]] inline F const* cm_data() const { return _p_CM; }

    /// CM - Column Major access to the Eigen Map
    [[nodiscard]] inline Eigen::Map<const EArrayCM> cm_emap() const { return _m_CM; }

  }; // End of class TSeries

} // End of namespace libtempo