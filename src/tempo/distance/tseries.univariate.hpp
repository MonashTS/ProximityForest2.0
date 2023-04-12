#pragma once

#include <tempo/dataset/tseries.hpp>
#include "univariate.hpp"

// Specialised implementation for TSeries

namespace tempo::distance::univariate {


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Elastic Distances
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /// ADTW with cost function cfe, penalty, and EAP cutoff.
  inline F adtw(TSeries const& series1, TSeries const& series2, F cfe, F penalty, F cutoff) {
    return adtw(series1.data(), series1.length(), series2.data(), series2.length(), cfe, penalty, cutoff);
  }

  /// DTW with cost function cfe, warping window length, and EAP cutoff.
  /// Use window=NO_WINDOW to use unconstrained DTW.
  inline F dtw(TSeries const& series1, TSeries const& series2, F cfe, size_t window, F cutoff) {
    return dtw(series1.data(), series1.length(), series2.data(), series2.length(), cfe, window, cutoff);
  }

  /// WDTW with cost function cfe, weights and EAP cutoff.
  inline F wdtw(TSeries const& series1, TSeries const& series2, F cfe, F const *weights, F cutoff) {
    return wdtw(series1.data(), series1.length(), series2.data(), series2.length(), cfe, weights, cutoff);
  }

  /// ERP with cost function cfe, gap value, warping window, and EAP cutoff.
  /// Use window=NO_WINDOW to use unconstrained ERP.
  inline F erp(TSeries const& series1, TSeries const& series2, F cfe, F gap_value, size_t window, F cutoff) {
    return erp(series1.data(), series1.length(), series2.data(), series2.length(), cfe, gap_value, window, cutoff);
  }

  /// LCSS with epsilon, warping window, and EAP cutoff.
  /// Use window=NO_WINDOW to use unconstrained LCSS.
  inline F lcss(TSeries const& series1, TSeries const& series2, F epsilon, size_t window, F cutoff) {
    return lcss(series1.data(), series1.length(), series2.data(), series2.length(), epsilon, window, cutoff);
  }

  /// MSM with cost and EAP cutoff.
  inline F msm(TSeries const& series1, TSeries const& series2, F cost, F cutoff) {
    return msm(series1.data(), series1.length(), series2.data(), series2.length(), cost, cutoff);
  }

  /// TWE with stiffness (nu) and penalty (lambda) parameters, and EAP cutoff.
  inline F twe(TSeries const& series1, TSeries const& series2, F nu, F lambda, F cutoff) {
    return twe(series1.data(), series1.length(), series2.data(), series2.length(), nu, lambda, cutoff);
  }


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // DTW Lower Bounds
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---


  /// LB Keogh for a query and a candidate represented by its envelopes.
  /// Only use for same length series. Tunable cost function cfe similar to dtw.
  inline F lb_Keogh(TSeries const& query, std::vector<F> const& upper, std::vector<F> const& lower, F cfe, F cutoff) {
    return lb_Keogh(query.data(), query.length(), upper, lower, cfe, cutoff);
  }

  /// LB Keogh 2 ways done 'jointly' - for same length series,
  /// with tunable cost function cfe similar to dtw
  inline F lb_Keogh2j(
    TSeries const& series1, std::vector<F> const& upper1, std::vector<F> const& lower1,
    TSeries const& series2, std::vector<F> const& upper2, std::vector<F> const& lower2,
    F cfe, F cutoff
  ) {
    return lb_Keogh2j(
      series1.data(), series1.length(), upper1, lower1,
      series2.data(), series2.length(), upper2, lower2,
      cfe, cutoff
    );
  }

  /// LB Enhanced for a query and a candidate series with its envelopes.
  /// Only use for same length series. Tunable cost function cfe similar to dtw.
  /// 'v' is the number of LR bands, speed/tightness trade-off (faster = 0, tighter = length/2)
  inline F lb_Enhanced(
    TSeries const& query, TSeries const& candidate,
    std::vector<F> const& candidate_up, std::vector<F> const& candidate_lo,
    F cfe, size_t v, size_t w, F cutoff
  ) {
    return lb_Enhanced(query.data(), query.length(), candidate.data(), candidate.length(),
                       candidate_up, candidate_lo, cfe, v, w, cutoff
    );
  }

  /// LB Enhanced 2 ways done 'jointly' for two series and their envelopes?
  /// Only use for same length series. Tunable cost function cfe similar to dtw.
  /// 'v' is the number of LR bands, speed/tightness trade-off (faster = 0, tighter = length/2)
  inline F lb_Enhanced2j(
    TSeries const& series1, std::vector<F> const& upper1, std::vector<F> const& lower1,
    TSeries const& series2, std::vector<F> const& upper2, std::vector<F> const& lower2,
    F cfe, size_t v, size_t w, F cutoff
  ) {
    return lb_Enhanced2j(series1.data(), series1.length(), upper1, lower1,
                         series2.data(), series2.length(), upper2, lower2,
                         cfe, v, w, cutoff
    );
  }

  /// LB Webb for two same length series sa and sb, with their envelopes,
  /// and the lower envelope of their upper envelopes,
  /// and the upper envelope of their lower envelopes.
  /// Tunable cost function cfe similar to dtw.
  inline F lb_Webb(
    // Series A
    TSeries const& a,
    std::vector<F> const& a_up, std::vector<F> const& a_lo,
    std::vector<F> const& a_lo_up, std::vector<F> const& a_up_lo,
    // Series B
    TSeries const& b,
    std::vector<F> const& b_up, std::vector<F> const& b_lo,
    std::vector<F> const& b_lo_up, std::vector<F> const& b_up_lo,
    // Others
    F cfe, size_t w, F cutoff
  ) {
    return lb_Webb(a.data(), a.length(), a_up, a_lo, a_lo_up, a_up_lo,
                   b.data(), b.length(), b_up, b_lo, b_lo_up, b_up_lo,
                   cfe, w, cutoff
    );
  }



  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Lockstep
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---


  /// Direct alignment with cost function cfe, and early abandoning cutoff.
  inline F directa(TSeries const& series1, TSeries const& series2, F cfe, F cutoff) {
    return directa<F>(series1.data(), series1.length(), series2.data(), series2.length(), cfe, cutoff);
  }

  /// Lorentzian metric, Armadillo vectorized, pointer interface
  inline F lorentzian(TSeries const& series1, TSeries const& series2) {
    return lorentzian(series1.rowvec(), series2.rowvec());
  }

  /// Minkowski metric, Armadillo vectorized, pointer interface
  ///  - Equal to the Manhattan distance with p=1
  ///  - Equal to the Euclidean Distance distance with p=2
  ///  - Also see the direct alignment function, which does the same without taking the root.
  ///    With exponent 0.5, 1, and 2, the direct alignment uses specialised cost function,
  ///    which may be faster, in particular for NN search.
  inline F minkowski(TSeries const& series1, TSeries const& series2, F p) {
    return minkowski(series1.rowvec(), series2.rowvec(), p);
  }

  /// Manhattan metric, Armadillo vectorized, pointer interface
  /// Special case for Minkowski with p=1
  inline F manhattan(TSeries const& series1, TSeries const& series2) {
    return manhattan(series1.rowvec(), series2.rowvec());
  }


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Sliding
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---


  /// SBD, Armadillo vectorized, pointer interface
  inline F sbd(TSeries const& series1, TSeries const& series2) {
    return sbd(series1.rowvec(), series2.rowvec());
  }

} // End of namespace tempo::distance::univariate
