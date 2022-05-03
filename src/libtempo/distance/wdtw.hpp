#pragma once

#include <libtempo/utils/utils.hpp>
#include "cost_function.hpp"
#include <vector>

namespace libtempo::distance {

  // --- --- --- --- ---
  // Weights generation
  // --- --- --- --- ---

  /// From the paper, changing this values does not change the results (scaling), so keep to 1
  constexpr double WDTW_MAX_WEIGHT = 1;

  /** Compute a weight at index i in a sequence 1..m
   * @param g "Controls the level of penalization for the points with larger phase difference".
   *        range [0, +inf), usually in [0.01, 0.6].
   *        Some examples:
   *        * 0: constant weight
   *        * 0.05: nearly linear weights
   *        * 0.25: sigmoid weights
   *        * 3: two distinct weights between half sequences
   *
   * @param half_max_length Mid point of the sequence (m/2)
   * @param i Index of the point in [1..m] (m=length of the sequence)
   * @return the weight for index i
   */
  template<typename FloatType>
  [[nodiscard]] inline FloatType compute_weight(FloatType g, FloatType half_max_length, FloatType i, double wmax) {
    return wmax/(1 + exp(-g*(i - half_max_length)));
  }

  /// Populate the weights_array of size length with weights derive from the g factor
  template<typename FloatType>
  inline void populate_weights(FloatType g, FloatType *weights_array, size_t length, double wmax = WDTW_MAX_WEIGHT) {
    FloatType half_max_length = FloatType(length)/2;
    for (size_t i{0}; i<length; ++i) {
      weights_array[i] = compute_weight(g, half_max_length, FloatType(i), wmax);
    }
  }

  /// Create a vector of weights
  template<typename FloatType>
  inline std::vector<FloatType> generate_weights(FloatType g, size_t length, double wmax = WDTW_MAX_WEIGHT) {
    std::vector<FloatType> weights(length, 0);
    populate_weights(g, weights.data(), length, wmax);
    return weights;
  }

  // --- --- --- --- ---
  // WDTW distance
  // --- --- --- --- ---

  namespace internal {

    /** Weighted Dynamic Time Warping with cutoff point for early abandoning and pruning.
     * Double buffered implementation using O(n) space.
     * Worst case scenario has a O(n²) time complexity (no pruning nor early abandoning).
     * A tight cutoff can allow a lot of pruning, speeding up the process considerably.
     * Actual implementation assuming that some pre-conditions are fulfilled.
     * @tparam FloatType    The floating number type used to represent the series.
     * @tparam D            Type of underlying collection - given to dist
     * @tparam FDist        Distance computation function, must be a (size_t, size_t)->FloatType
     * @tparam VecLike      Vector like datatype - type of "weights", accessed with [index]
     * @param nblines       Length of the line series. Must be 0 < nbcols <= nblines
     * @param nbcols        Length of the column series. Must be 0 < nbcols <= nblines
     * @param dist          Distance function of type FDist
     * @param cutoff.       Attempt to prune computation of alignments with cost > cutoff.
     *                      May lead to early abandoning.
     * @return WDTW between the two series or +INF if early abandoned.
     */
    template<Float F>
    [[nodiscard]] inline F wdtw(
      const size_t nblines,
      const size_t nbcols,
      CFun <F> auto dist,
      const Subscriptable auto& weights,
      F cutoff,
      std::vector<F>& buffers_v
    ) {
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // In debug mode, check preconditions
      assert(nblines!=0);
      assert(nbcols!=0);
      // Adapt constants to the floating point type
      using namespace utils;
      constexpr auto PINF = utils::PINF<F>;

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Create a new tighter upper bounds (most commonly used in the code).
      // First, take the "next float" after "cutoff" to deal with numerical instability.
      // Then, subtract the cost of the last alignment.
      const F ub = initBlock {
        return nextafter(cutoff, PINF) - dist(nblines - 1, nbcols - 1)*weights[0];
      };

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Double buffer allocation, init to +INF.
      // Base indices for the 'c'urrent row and the 'p'revious row. Account for the extra cell (+1 and +2)
      // std::vector<FloatType> buffers_v((1+nbcols)*2, PINF);
      // auto* buffers = buffers_v.data();
      buffers_v.assign((1 + nbcols)*2, PINF);
      auto *buffers = buffers_v.data();
      size_t c{0 + 1}, p{nbcols + 2};

      // Line & column counters
      size_t i{0}, j{0};

      // Cost accumulator. Also used as the "left neighbour".
      double cost{0};

      // EAP variables: track where to start the next line, and the position of the previous pruning point.
      // Must be init to 0: index 0 is the next starting index and also the "previous pruning point"
      size_t next_start{0}, prev_pp{0};

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Initialisation of the top border: already initialized to +INF. Initialise the left corner to 0.
      buffers[c - 1] = 0;

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Main loop
      for (; i<nblines; ++i) {
        // --- --- --- Swap and variables init
        std::swap(c, p);
        size_t curr_pp = next_start; // Next pruning point init at the start of the line
        j = next_start;
        // --- --- --- Stage 0: Initialise the left border
        {
          cost = PINF;
          buffers[c + next_start - 1] = PINF;
        }
        // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
        for (; j==next_start&&j<prev_pp; ++j) {
          const auto d = dist(i, j)*weights[utils::absdiff(i, j)];
          cost = std::min(buffers[p + j - 1], buffers[p + j]) + d;
          buffers[c + j] = cost;
          if (cost<=ub) { curr_pp = j + 1; } else { ++next_start; }
        }
        // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
        for (; j<prev_pp; ++j) {
          const auto d = dist(i, j)*weights[utils::absdiff(i, j)];
          cost = min(cost, buffers[p + j - 1], buffers[p + j]) + d;
          buffers[c + j] = cost;
          if (cost<=ub) { curr_pp = j + 1; }
        }
        // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
        if (j<nbcols) { // If so, two cases.
          const auto d = dist(i, j)*weights[utils::absdiff(i, j)];
          if (j==next_start) { // Case 1: Advancing next start: only diag.
            cost = buffers[p + j - 1] + d;
            buffers[c + j] = cost;
            if (cost<=ub) { curr_pp = j + 1; }
            else {
              // Special case if we are on the last alignment: return the actual cost if we are <= cutoff
              if (i==nblines - 1&&j==nbcols - 1&&cost<=cutoff) { return cost; }
              else { return PINF; }
            }
          } else { // Case 2: Not advancing next start: possible path in previous cells: left and diag.
            cost = std::min(cost, buffers[p + j - 1]) + d;
            buffers[c + j] = cost;
            if (cost<=ub) { curr_pp = j + 1; }
          }
          ++j;
        } else { // Previous pruning point is out of bound: exit if we extended next start up to here.
          if (j==next_start) {
            // But only if we are above the original UB
            // Else set the next starting point to the last valid column
            if (cost>cutoff) { return PINF; }
            else { next_start = nbcols - 1; }
          }
        }
        // --- --- --- Stage 4: After the previous pruning point: only prev.
        // Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
        for (; j==curr_pp&&j<nbcols; ++j) {
          const auto d = dist(i, j)*weights[utils::absdiff(i, j)];
          cost = cost + d;
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
   * @tparam FloatType  The floating number type used to represent the series.
   * @tparam D          Type of underlying collection - given to dist
   * @tparam VecLike    Vector like datatype - type of "weights", accessed with [index]
   * @tparam FDist      Distance computation function, must be a (size_t, size_t)->FloatType
   * @param length1     Length of the first series.
   * @param length2     Length of the second series.
   * @param dist        Distance function of type FDist
   * @param weights     Pointer to the weights. Must be at least as long as nblines.
   * @param ub          Upper bound. Attempt to prune computation of alignments with cost > cutoff.
   *                    May lead to early abandoning.
   *                    If not provided, defaults to PINF,
   *                    which triggers the computation of an upper bound based on the diagonal.
   *                    Use QNAN to run without any upper bounding.
   * @return DTW between the two series
   */
  template<Float F>
  [[nodiscard]]
  F wdtw(size_t nblines, size_t nbcols,
         CFun <F> auto dist,
         const Subscriptable auto& weights,
         F ub,
         std::vector<F>& buffer_v
  ) {
    constexpr F INF = utils::PINF<F>;
    if (nblines==0&&nbcols==0) { return 0; }
    else if ((nblines==0)!=(nbcols==0)) { return INF; }
    else {
      // Compute a cutoff point using the diagonal
      if (std::isinf(ub)) {
        ub = 0;
        // Cover diagonal
        const auto m = std::min(nblines, nbcols);
        for (size_t i{0}; i<m; ++i) { ub = ub + dist(i, i)*weights[0]; }
        // Fewer line than columns: complete the last line (advance in the columns)
        if (nblines<nbcols) {
          for (size_t j{nblines}; j<nbcols; ++j) {
            ub = ub + dist(nblines - 1, j)*weights[utils::absdiff(nblines - 1, j)];
          }
        }
          // Fewer columns than lines: complete the last column (advance in the lines)
        else if (nbcols<nblines) {
          for (size_t i{nbcols}; i<nblines; ++i) {
            ub = ub + dist(i, nbcols - 1)*weights[utils::absdiff(i, nbcols - 1)];
          }
        }
      } else if (std::isnan(ub)) { ub = INF; }
      // ub computed
      return internal::wdtw<F>(nblines, nbcols, dist, weights, ub, buffer_v);
    }
  }

  /// Helper without having to provide a buffer
  template<Float F>
  [[nodiscard]] inline F wdtw(size_t nblines, size_t nbcols,
                              CFun <F> auto dist,
                              const Subscriptable auto& weights,
                              F ub) {
    std::vector<F> v;
    return wdtw(nblines, nbcols, dist, weights, ub, v);
  }

  /// Helper for TSLike, without having to provide a buffer
  template<Float F, TSLike T>
  [[nodiscard]] inline F
  wdtw(const T& lines,
       const T& cols,
       CFunBuilder<T> auto mkdist,
       const std::vector<F>& weights,
       F ub = utils::PINF<F>) {
    const auto ls = lines.length();
    const auto cs = cols.length();
    const CFun <F> auto dist = mkdist(lines, cols);
    std::vector<F> v;
    return wdtw<F>(ls, cs, dist, weights, ub, v);
  }


} // End of namespace libtempo::distance
