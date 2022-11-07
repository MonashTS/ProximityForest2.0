#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/distance__/helpers.hpp>

namespace tempo::distance {

  /// TWE specific cost function concept, computing the cost when taking a **non diagonal**
  /// step (warping the alignments). All required information must be captured.
  template<typename Fun>
  concept CFunTWE = requires(Fun fun, size_t i){
    { fun(i) }->std::convertible_to<F>;
  };

  /// CFunTWEBuilder: Function creating a CFunTWE based on 1 series, nu, and lambda
  /// Use for warping step (cost matrix: step takes line or col)
  template<typename T, typename D>
  concept CFunWarpTWEBuilder = requires(T builder, const D& s, const F nu, const F lambda){
    builder(s, nu, lambda);
  };

  /// CFunDiagTWEBuilder: Function creating a CFunTWE based on 2 series (line, column), nu, and lambda
  /// Use for non warping step (cost matrix: set takes diagonal)
  template<typename T, typename D>
  concept CFunDiagTWEBuilder = requires(T builder, const D& lines, const D& cols, const F nu){
    builder(lines, cols, nu);
  };

  namespace internal {

    /** Time Warp Edit (TWE), with cut-off point for early abandoning and pruning.
     *  Double buffered implementation using O(n) space.
     *  Worst case scenario has a O(n²) time complexity (no pruning nor early abandoning, large window).
     *  A tight cutoff can allow a lot of pruning, speeding up the process considerably.
     *  Actual implementation assuming that some pre-conditions are fulfilled.
     * @param lines         Data for the lines
     * @param nblines       Length of the line series. Must be 0 < nbcols <= nblines
     * @param cols          Data for the lines
     * @param nbcols        Length of the column series. Must be 0 < nbcols <= nblines
     * @param dist_diag     Diag step function of type FDist
     * @param nu            Stiffness parameter
     * @param cutoff        Attempt to prune computation of alignments with cost > cutoff.
     *                      May lead to early abandoning.
     * @return TWE value or +INF if early abandoned, or , given w, no alignment is possible
     */
    [[nodiscard]] inline F twe(const size_t nblines,
                               const size_t nbcols,
                               CFunTWE auto dist_lines,
                               CFunTWE auto dist_cols,
                               CFun auto dist,
                               const F nu,
                               const F cutoff,
                               std::vector<F>& buffer_v
    ) {
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // In debug mode, check preconditions
      assert(nblines!=0);
      assert(nbcols!=0);
      // Adapt constants to the floating point type
      using namespace utils;
      using utils::PINF;

      //  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      //  // Constants: we only consider timestamp spaced by 1, so:
      //  // In the "delete" case, we always have a time difference of 1, so we always have 1*nu+lambda
      //  const auto nu_lambda = nu+lambda;
      //  // In the "match" case, we always have nu*(|i-j|+|(i-1)-(j-1)|) == 2*nu*|i-j|
      const auto nu2_ = F(2)*nu;
      const auto dist_diag = [&](size_t i, size_t j) {
        const F da = dist(i, j);
        const F db = dist(i - 1, j - 1);
        return da + db + nu2_*absdiff(i, j);
      };

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Create a new tighter upper bounds (most commonly used in the code).
      // First, take the "next float" after "cutoff" to deal with numerical instability.
      // Then, subtract the cost of the last alignment.
      const F ub = initBlock {
        // The last alignment can only computed if we have nbcols >= 2
        if (nbcols>=2) {
          const auto la = min(
            // "Delete_B": over the columns / Prev
            // dist(cols, nbcols-2, cols, nbcols-1)+nu_lambda --> Capture in dist_cols
            dist_cols(nbcols - 1),
            // Match: Diag. Ok: nblines >= nbcols
            // dist(lines, nblines-1, cols, nbcols-1)+dist(lines, nblines-2, cols, nbcols-2)+nu2d(nblines-nbcols),
            dist_diag(nblines - 1, nbcols - 1),
            // "Delete_A": over the lines / Top
            // dist(lines, nblines-2, lines, nblines-1)+nu_lambda --> Capture in dist_lines
            dist_lines(nblines - 1)
          );
          return F(nextafter(cutoff, PINF) - la);
        } else { return F(cutoff); }
      };

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Double buffer allocation, no initialisation required (border condition manage in the code).
      // Base indices for the 'c'urrent row and the 'p'revious row.
      //auto buffers = std::unique_ptr<F[]>(new F[nbcols*2]);
      buffer_v.assign(nbcols*2, 0);
      auto *buffers = buffer_v.data();
      size_t c{0}, p{nbcols};

      // Buffer holding precomputed distance between columns
      auto distcol = std::unique_ptr<F[]>(new F[nbcols]);

      // Line & column counters
      size_t i{0}, j{0};

      // Cost accumulator. Also used as the "left neighbour".
      F cost{0};

      // EAP variables: track where to start the next line, and the position of the previous pruning point.
      // Must be init to 0: index 0 is the next starting index and also the "previous pruning point"
      size_t next_start{0}, prev_pp{0};

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Initialisation of the first line. Deal with the line top border condition.
      {
        // Case [0,0]: special "Match case"
        cost = dist(0, 0);
        buffers[c + 0] = cost;
        // Distance for the first column is relative to 0 "by conventions" (from the paper, section 4.2)
        // Something like
        // distcol[0]=dist(0, cols[0]);
        // However, this would only be use for j=0, i.e. with the left border which is +INF,
        // hence the result would always be +INF: we can simply use 0 instead.
        // Note that the border is managed as part of the code, hence we actually never access distcol[0]!
        // distcol[0] = 0;
        // Rest of the line: [i==0, j>=1]: "Delete_B case" (prev)
        // We also initialize 'distcol' here.
        for (j = 1; j<nbcols; ++j) {
          const double d = dist_cols(j);
          distcol[j] = d;
          cost = cost + d;
          buffers[c + j] = cost;
          if (cost<=ub) { prev_pp = j + 1; } else { break; }
        }
        // Complete the initialisation of distcol
        for (; j<nbcols; ++j) { distcol[j] = dist_cols(j); }
        // Next line.
        ++i;
      }


      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Main loop, starts at the second line
      for (; i<nblines; ++i) {
        // --- --- --- Swap and variables init
        std::swap(c, p);
        const double distli = dist_lines(i);
        size_t curr_pp = next_start; // Next pruning point init at the start of the line
        j = next_start;
        // --- --- --- Stage 0: Special case for the first column. Can only look up (border on the left)
        {
          cost = buffers[p + j] + distli; // "Delete_A" / Top
          buffers[c + j] = cost;
          if (cost<=ub) { curr_pp = j + 1; } else { ++next_start; }
          ++j;
        }
        // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
        for (; j==next_start&&j<prev_pp; ++j) {
          cost = std::min(
            buffers[p + j - 1] + dist_diag(i, j),     // "Match" / Diag
            buffers[p + j] + distli                   // "Delete_A" / Top
          );
          buffers[c + j] = cost;
          if (cost<=ub) { curr_pp = j + 1; } else { ++next_start; }
        }
        // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
        for (; j<prev_pp; ++j) {
          cost = min(
            cost + distcol[j],                      // "Delete_B": over the columns / Prev
            buffers[p + j - 1] + dist_diag(i, j),   // Match: Diag
            buffers[p + j] + distli                 // "Delete_A": over the lines / Top
          );
          buffers[c + j] = cost;
          if (cost<=ub) { curr_pp = j + 1; }
        }
        // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
        if (j<nbcols) { // If so, two cases.
          if (j==next_start) { // Case 1: Advancing next start: only diag.
            cost = buffers[p + j - 1] + dist_diag(i, j); // Match: Diag
            buffers[c + j] = cost;
            if (cost<=ub) { curr_pp = j + 1; }
            else {
              // Special case if we are on the last alignment: return the actual cost if we are <= cutoff
              if (i==nblines - 1&&j==nbcols - 1&&cost<=cutoff) { return cost; } else { return PINF; }
            }
          } else { // Case 2: Not advancing next start: possible path in previous cells: left and diag.
            cost = std::min(
              cost + distcol[j],                      // "Delete_B": over the columns / Prev
              buffers[p + j - 1] + dist_diag(i, j)    // Match: Diag
            );
            buffers[c + j] = cost;
            if (cost<=ub) { curr_pp = j + 1; }
          }
          ++j;
        } else { // Previous pruning point is out of bound: exit if we extended next start up to here.
          if (j==next_start) {
            // But only if we are above the original UB. Else set the next starting point to the last valid column
            if (cost>cutoff) { return PINF; } else { next_start = nbcols - 1; }
          }
        }
        // --- --- --- Stage 4: After the previous pruning point: only prev.
        // Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
        for (; j==curr_pp&&j<nbcols; ++j) {
          cost = cost + distcol[j]; // "Delete_B": over the columns / Prev
          buffers[c + j] = cost;
          if (cost<=ub) { ++curr_pp; }
        }
        // --- --- ---
        prev_pp = curr_pp;
      } // End of main loop for(;i<nblines;++i)

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Finalisation
      // Check for last alignment (i==nblines implied, Stage 4 implies j<=nbcols). Cost must be <= original bound.
      if (j==nbcols&&cost<=cutoff) { return cost; } else { return PINF; }
    }

  } // End of namespace internal


  /** Early Abandoned and Pruned Dynamic Time Warping.
   * @param length1     Length of the first series.
   * @param length2     Length of the second series.
   * @param dist        Distance function of type FDist
   * @param nu          Stiffness parameter
   * @param ub          Upper bound. Attempt to prune computation of alignments with cost > cutoff.
   *                    May lead to early abandoning.
   *                    If not provided, defaults to PINF,
   *                    which triggers the computation of an upper bound based on the diagonal.
   *                    Use QNAN to run without any upper bounding.
   * @return DTW between the two series
   */
  [[nodiscard]] F inline twe(const size_t nblines,
                             const size_t nbcols,
                             CFunTWE auto dist_lines,
                             CFunTWE auto dist_cols,
                             CFun auto dist,
                             const F nu,
                             F ub,
                             std::vector<F>& buffer_v
  ) {
    constexpr F INF = utils::PINF;
    if (nblines==0&&nbcols==0) { return 0; }
    else if ((nblines==0)!=(nbcols==0)) { return INF; }
    else {
      // Compute a cutoff point using the diagonal
      if (std::isinf(ub)) {
        const auto m = std::min(nblines, nbcols);
        ub = 0;
        // Init case
        ub = dist(0, 0);
        // Diagonal
        const auto nu2_ = F(2)*nu;
        const auto dist_diag = [&](size_t i, size_t j) {
          const F da = dist(i, j);
          const F db = dist(i - 1, j - 1);
          return da + db + nu2_*tempo::utils::absdiff(i, j);
        };
        // Cover diagonal
        for (size_t i{1}; i<m; ++i) { ub = ub + dist_diag(i, i); }
        // Fewer line than columns: complete the last line (advancing in the columns)
        if (nblines<nbcols) {
          for (size_t j{nblines}; j<nbcols; ++j) { ub = ub + dist_cols(j); }
        }
          // Fewer columns than lines: complete the last column (advancing in the lines)
        else if (nbcols<nblines) {
          for (size_t i{nbcols}; i<nblines; ++i) { ub = ub + dist_lines(i); }
        }
      } else if (std::isnan(ub)) { ub = INF; }
      // ub computed
      return internal::twe(nblines, nbcols, dist_lines, dist_cols, dist, nu, ub, buffer_v);
    }
  }

  // Helper without having to provide a buffer
  [[nodiscard]] inline F twe(const size_t nblines,
                             const size_t nbcols,
                             CFunTWE auto dist_lines,
                             CFunTWE auto dist_cols,
                             CFun auto dist,
                             F nu,
                             F ub) {
    std::vector<F> v;
    const F r = twe(nblines, nbcols, dist_lines, dist_cols, dist, nu, ub, v);
    return r;
  }

  /// Helper for TSLike, without having to provide a buffer
  template<TSLike T>
  [[nodiscard]] inline F twe(const T& lines,
                             const T& cols,
                             CFunWarpTWEBuilder<T> auto mkdist_lines,
                             CFunWarpTWEBuilder<T> auto mkdist_cols,
                             CFunBuilder<T> auto mkdist_base,
                             const F nu, const F lambda,
                             F ub) {
    const auto ls = lines.length();
    const auto cs = cols.length();
    const CFunTWE auto dist_lines = mkdist_lines(lines, nu, lambda);
    const CFunTWE auto dists_cols = mkdist_cols(cols, nu, lambda);
    const CFun auto dist_base = mkdist_base(lines, cols);
    return twe(ls, cs, dist_lines, dists_cols, dist_base, nu, ub);
  }

  namespace univariate {

    /// Default TWE warping step cost function, using univariate ad2
    template<Subscriptable D>
    [[nodiscard]] inline auto twe_warp_ad2(const D& s, const F nu, const F lambda) {
      const F nl = nu + lambda;
      return [&, nl](size_t i) {
        const auto d = s[i] - s[i - 1];
        return (d*d) + nl;
      };
    }

    /// Default TWE using univariate ad2
    template<TSLike T>
    [[nodiscard]] inline F twe(const T& lines, const T& cols, const F nu, const F lambda, F ub) {
      return tempo::distance::twe<T>(lines, cols,
                                     twe_warp_ad2<T>, twe_warp_ad2<T>, ad2<T>,
                                     nu, lambda, ub
      );
    }

    /// Specific overload for univariate vector
    [[nodiscard]] inline F twe(const std::vector<F>& lines,
                               const std::vector<F>& cols,
                               CFunWarpTWEBuilder<std::vector<F>> auto mkdist_lines,
                               CFunWarpTWEBuilder<std::vector<F>> auto mkdist_cols,
                               CFunBuilder<std::vector<F>> auto mkdist_base,
                               const F nu,
                               const F lambda,
                               F ub) {
      const auto ls = lines.size();
      const auto cs = cols.size();
      const CFunTWE auto dist_lines = mkdist_lines(lines, nu, lambda);
      const CFunTWE auto dists_cols = mkdist_cols(cols, nu, lambda);
      const CFun auto dist_base = mkdist_base(lines, cols);
      return tempo::distance::twe(ls, cs, dist_lines, dists_cols, dist_base, nu, ub);
    }

    /// Specific overload for univariate vector using ad2
    [[nodiscard]] inline F twe(const std::vector<F>& lines,
                               const std::vector<F>& cols,
                               const F nu,
                               const F lambda,
                               F ub) {
      return twe(lines, cols, twe_warp_ad2<std::vector<F>>, twe_warp_ad2<std::vector<F>>, ad2<std::vector<F>>, nu,
                 lambda, ub
      );
    }
  }

  namespace multivariate {

    /// Default TWE warping step cost function, using multivariate ad2
    template<Subscriptable D>
    [[nodiscard]] inline auto twe_warp_ad2(const D& s, size_t ndim, const F nu, const F lambda) {
      const F nl = nu + lambda;
      return [&, ndim, nl](size_t i) { return ad2N<D>(s, s, ndim)(i, i - 1) + nl; };
    }

    /// Default TWE diagonal step cost function, using multivariate ad2
    template<Subscriptable D>
    [[nodiscard]] inline auto twe_diag_ad2(const D& lines, const D& cols, size_t ndim, const F nu) {
      const F nu2 = nu*2;
      return [&, ndim, nu2](size_t i, size_t j) {
        const auto da = ad2N<D>(lines, cols, ndim)(i, j);
        const auto db = ad2N<D>(lines, cols, ndim)(i - 1, j - 1);
        const auto r = da + db + (nu2*utils::absdiff(i, j));
        return r;
      };
    }

  }

} // End of namespace tempo::distance
