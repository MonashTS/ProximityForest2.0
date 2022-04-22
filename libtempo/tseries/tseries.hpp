#pragma once

#include <libtempo/utils/capsule.hpp>
#include <libtempo/concepts.hpp>

#include <armadillo>

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
    using MAT = arma::Mat<F>;

    /// Missing data in the time series? We use a floating point type, so should be represented by "nan"
    bool _missing{false};

    /// Optional label
    std::optional<L> _olabel{};

    /// Capsule: used when we own the data
    lu::Capsule _capsule;

    /// Raw Data pointer: the matrix must be built on top of that. Must be Column Major.
    F const *_rawdata{nullptr};

    /// Representation of the matrix - by default, 1 line (univariate), 0 cols (empty)
    MAT _matrix{1, 0};

    /// Private "moving-in" constructor
    TSeries(
      // Column major data
      lu::Capsule&& c,
      F const *p,
      MAT&& m,
      // Other
      std::optional<L> olabel,
      bool has_missing
    ) :
      _missing(has_missing),
      _olabel(olabel),
      _capsule(std::move(c)),
      _rawdata(p),
      _matrix(std::move(m)) {}

  public:

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Construction
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /** Default constructor: create an univaraite empty time series */
    TSeries() = default;

    /** Default move constructor */
    TSeries(TSeries&&) noexcept = default;

    /** Disable copy (we have to move instead; prevent unwanted data duplication) */
    TSeries(const TSeries&) = delete;

    /** Build a new series from a row major vector */
    [[nodiscard]] static TSeries mk_rowmajor(
      std::vector<F>&& v,
      size_t nbvar,
      std::optional<L> olabel,
      std::optional<bool> omissing
    ) {
      using namespace std;

      // --- Checking
      size_t vsize = v.size();
      if (nbvar<1) { throw domain_error("Number of variable can't be < 1"); }
      if (vsize%nbvar!=0) { throw domain_error("Vector size is not a multiple of 'nbvar'"); }

      // --- lico
      const size_t nb_lines = nbvar;
      const size_t nb_cols = vsize/nbvar;

      // --- Take ownership of the incoming vector
      auto capsule = lu::make_capsule<vector<F>>(move(v));

      // --- Build Armadillo matrix
      // Armadillo works with column major data, but we are given a row major one.
      // Build the data, and proceed with an "in place" transposition
      // This transposition **will** change the underlying vector, which is fine.
      F *rawptr = lu::get_capsule_ptr<vector<F>>(capsule)->data();
      // Invert line/column here: we get the "right" matrix after transposition
      MAT matrix(rawptr, nb_cols, nb_lines,
                 false,   // copy_aux_mem = false: use the auxiliary memory (i.e. no copying)
                 true     // struct = true: matrix bounds to the auxiliary memory for its lifetime; can't be resized
      );

      // Transpose
      inplace_trans(matrix);
      assert(nb_lines==matrix.n_rows);
      assert(nb_cols==matrix.n_cols);

      // Check missing data (NAN)
      bool has_missing;
      if (omissing.has_value()) { has_missing = omissing.value(); }
      else { has_missing = matrix.has_nan(); }

      // Build the TSeries
      return TSeries(
        std::move(capsule),
        rawptr,
        std::move(matrix),
        // Other
        olabel,
        has_missing
      );
    }

    /// Copy data from other, except for the actual data. Allow to easily do transforms. No checking done.
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
    [[nodiscard]] inline size_t nvar() const { return _matrix.n_rows; }

    /// Access the length
    [[nodiscard]] inline size_t length() const { return _matrix.n_cols; }

    /// Access the size of the data == ndim*length
    [[nodiscard]] inline size_t size() const { return _matrix.n_elem; }

    /// Check if has missing values
    [[nodiscard]] inline bool missing() const { return _missing; }

    /// Get the label (perform a copy)
    [[nodiscard]] inline std::optional<L> label() const { return _olabel; }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Data access
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Column major access over all the points AND dimensions
    [[nodiscard]] inline F operator [](size_t idx) const { return _matrix(idx); }

    /// Column major access to the raw pointer
    [[nodiscard]] inline const F* rawdata() const { return _rawdata; }

    /// Matrix access (li, co)
    [[nodiscard]] inline const MAT& data() const { return _matrix; }



  }; // End of class TSeries

} // End of namespace libtempo