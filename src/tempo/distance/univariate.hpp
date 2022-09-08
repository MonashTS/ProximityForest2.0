#pragma once

#include <vector>
#include "utils.hpp"

namespace tempo::distance::univariate {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Elastic Distances
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /// ADTW with cost function cfe, penalty, and EAP cutoff.
  template<typename F>
  F adtw(
    F const *data1, size_t length1,
    F const *data2, size_t length2,
    F cfe,
    F penalty,
    F cutoff
  );

  /// DTW with cost function cfe, warping window length, and EAP cutoff.
  /// Use window=NO_WINDOW to use unconstrained DTW.
  template<typename F>
  F dtw(
    F const *data1, size_t length1,
    F const *data2, size_t length2,
    F cfe,
    size_t window,
    F cutoff
  );

  /// WDTW with cost function cfe, weights and EAP cutoff.
  template<typename F>
  F wdtw(F const *data1, size_t length1,
         F const *data2, size_t length2,
         F cfe,
         F const *weights,
         F cutoff
  );

  /// Populate a pointed array of size length by weights suitable for WDTW.
  template<typename F>
  void wdtw_weights(F g, F *weights_array, size_t length, F wmax = 1);

  /** Generate a vector of weights
   * @tparam F              Floating type used for the computation
   * @param g               Level of penalisation - see 'compute_weight'
   * @param weights         Output vector - reallocation may occur if weights.size() < length
   * @param length          Should be the maximum possible length of a series
   * @param wmax            Scaling - default to 1
   * @return                A vector of weight suitable for WDTW
   */
  template<typename F>
  void wdtw_weights(F g, std::vector<F>& weights, size_t length, F wmax = 1);

  /** Same as above, creating the vector instead of reusing one. */
  template<typename F>
  std::vector<F> wdtw_weights(F g, size_t length, F wmax=1);

  /// ERP with cost function cfe, gap value, warping window, and EAP cutoff.
  /// Use window=NO_WINDOW to use unconstrained ERP.
  template<typename F>
  F erp(F const *data1, size_t length1,
        F const *data2, size_t length2,
        F cfe,
        F gap_value,
        size_t window,
        F cutoff
  );

  /// LCSS with epsilon, warping window, and EAP cutoff.
  /// Use window=NO_WINDOW to use unconstrained LCSS.
  template<typename F>
  F lcss(F const *data1, size_t length1,
         F const *data2, size_t length2,
         F epsilon,
         size_t window,
         F cutoff
  );

  /// MSM with cost and EAP cutoff.
  template<typename F>
  F msm(F const *data1, size_t length1,
        F const *data2, size_t length2,
        F cost,
        F cutoff
  );

  /// TWE with stiffness (nu) and penalty (lambda) parameters, and EAP cutoff.
  template<typename F>
  F twe(F const *data1, size_t length1,
        F const *data2, size_t length2,
        F nu,
        F lambda,
        F cutoff
  );



  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // DTW Lower Bounds
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  // --- --- --- Envelopes
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /// Given a series, compute both the upper and lower envelopes for a window w
  /// Write the results in upper and lower (must point to buffers of size >= length)
  template<typename F>
  void get_keogh_envelopes(F const *series, size_t length, F *upper, F *lower, size_t w);

  /** Compute the upper and lower envelopes of a series, suitable for LB_Keogh.
   *  Wrapper for get_envelopes with vector
   * @tparam F      Floating type used for the computation
   * @param series  Input series
   * @param length  Length of the input series
   * @param upper   Output series - reallocation may occur!
   * @param lower   Output series - reallocation may occur!
   * @param w       The window for which the envelope is computed.
   */
  template<typename F>
  void get_keogh_envelopes(F const *series, size_t length, std::vector<F>& upper, std::vector<F>& lower, size_t w);

  /// Given a series, compute the upper envelope for a window w
  /// Write the results in upper (must point to a buffer of size >= length)
  template<typename F>
  void get_keogh_up_envelope(F const *series, size_t length, F *upper, size_t w);

  /** Compute only the upper envelopes of a series.
   *  Wrapper for get_envelopes with vector
   * @tparam F      Floating type used for the computation
   * @param series  Input series
   * @param length  Length of the input series
   * @param upper   Output series - reallocation may occur!
   * @param w       The window for which the envelope is computed.
   */
  template<typename F>
  void get_keogh_up_envelope(F const *series, size_t length, std::vector<F>& upper, size_t w);

  /// Given a series, compute the lower envelope for a window w
  /// Write the results in lower (must point to a buffer of size >= length)
  template<typename F>
  void get_keogh_lo_envelope(F const *series, size_t length, F *lower, size_t w);

  /** Compute the lower envelopes of a series.
   *  Wrapper for get_envelopes with vector
   * @tparam F      Floating type used for the computation
   * @param series  Input series
   * @param length  Length of the input series
   * @param lower   Output series - reallocation may occur!
   * @param w       The window for which the envelope is computed.
   */
  template<typename F>
  void get_keogh_lo_envelope(F const *series, size_t length, std::vector<F>& lower, size_t w);

  /// Given a series, compute all the envelopes require for lb_Webb.
  /// Write the results in upper, lower, lower_upper and upper_lower, which must point to large enough buffers.
  template<typename F>
  void get_keogh_envelopes_Webb(
    F const *series, size_t length, F *upper, F *lower, F *lower_upper, F *upper_lower, size_t w
  );

  /** Compute all the envelopes requied by LB Webb for a series
   *  Reallocation may occur!
   * @tparam F          Floating point used for the computation
   * @param series      Input series
   * @param length      Length of the input series
   * @param upper       Output envelope
   * @param lower       Output envelope
   * @param lower_upper Output envelope
   * @param upper_lower Output envelope
   * @param w           Warping window
   */
  template<typename F>
  void get_keogh_envelopes_Webb(
    F const *series, size_t length, std::vector<F>& upper, std::vector<F>& lower,
    std::vector<F>& lower_upper, std::vector<F>& upper_lower,
    size_t w
  );



  // --- --- --- Lower Bounds
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---


  /// LB Keogh for a query and a candidate represented by its envelopes.
  /// Only use for same length series. Tunable cost function cfe similar to dtw.
  template<typename F>
  F lb_Keogh(F const *query, size_t query_length, F const *upper, F const *lower, F cfe, F cutoff);

  template<typename F>
  F lb_Keogh(F const *query, size_t lquery, std::vector<F> const& upper, std::vector<F> const& lower, F cfe, F cutoff);

  //

  /// LB Keogh 2 ways done 'jointly' - for same length series,
  /// with tunable cost function cfe similar to dtw
  template<typename F>
  F lb_Keogh2j(
    F const *series1, size_t length1, F const *upper1, F const *lower1,
    F const *series2, size_t length2, F const *upper2, F const *lower2,
    F cfe, F cutoff
  );

  template<typename F>
  F lb_Keogh2j(
    F const *series1, size_t length1, std::vector<F> const& upper1, std::vector<F> const& lower1,
    F const *series2, size_t length2, std::vector<F> const& upper2, std::vector<F> const& lower2,
    F cfe, F cutoff
  );

  //

  /// LB Enhanced for a query and a candidate series with its envelopes.
  /// Only use for same length series. Tunable cost function cfe similar to dtw.
  /// 'v' is the number of LR bands, speed/tightness trade-off (faster = 0, tighter = length/2)
  template<typename F>
  F lb_Enhanced(const F *query, size_t lquery,
                const F *candidate, size_t lcandidate, const F *candidate_upper, const F *candidate_lower,
                F cfe, size_t v, size_t w, F cutoff);

  template<typename F>
  F lb_Enhanced(const F *query, size_t lquery,
                const F *candidate, size_t lcandidate,
                std::vector<F> const& candidate_up, std::vector<F> const& candidate_lo,
                F cfe, size_t v, size_t w, F cutoff
  );

  //

  /// LB Enhanced 2 ways done 'jointly' for two series and their envelopes?
  /// Only use for same length series. Tunable cost function cfe similar to dtw.
  /// 'v' is the number of LR bands, speed/tightness trade-off (faster = 0, tighter = length/2)
  template<typename F>
  F lb_Enhanced2j(
    const F *series1, size_t length1, const F *upper1, const F *lower1,
    const F *series2, size_t length2, const F *upper2, const F *lower2,
    F cfe, size_t v, size_t w, F cutoff
  );

  template<typename F>
  F lb_Enhanced2j(
    const F *series1, size_t length1, std::vector<F> const& upper1, std::vector<F> const& lower1,
    const F *series2, size_t length2, std::vector<F> const& upper2, std::vector<F> const& lower2,
    F cfe, size_t v, size_t w, F cutoff
  );

  //

  /// LB Webb for two same length series sa and sb, with their envelopes,
  /// and the lower envelope of their upper envelopes,
  /// and the upper envelope of their lower envelopes.
  /// Tunable cost function cfe similar to dtw.
  template<typename F>
  F lb_Webb(
    // Series A
    F const *sa, size_t length_sa,
    F const *upper_sa, F const *lower_sa,
    F const *lower_upper_sa, F const *upper_lower_sa,
    // Series B
    F const *sb, size_t length_sb,
    F const *upper_sb, F const *lower_sb,
    F const *lower_upper_sb, F const *upper_lower_sb,
    // Others
    F cfe, size_t w, F cf
  );

  template<typename F>
  F lb_Webb(
    // Series A
    F const *a, size_t a_len,
    std::vector<F> const& a_up, std::vector<F> const& a_lo,
    std::vector<F> const& a_lo_up, std::vector<F> const& a_up_lo,
    // Series B
    F const *b, size_t b_len,
    std::vector<F> const& b_up, std::vector<F> const& b_lo,
    std::vector<F> const& b_lo_up, std::vector<F> const& b_up_lo,
    // Others
    F cfe, size_t w, F cutoff
  );






  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Lockstep
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /// Direct alignment with cost function cfe, and early abandoning cutoff.
  template<typename F>
  F directa(F const *data1, size_t length1, F const *data2, size_t length2, F cfe, F cutoff);

  /// Lorentzian metric, Armadillo vectorized, pointer interface
  template<typename F>
  F lorentzian(F const *A, size_t lA, F const *B, size_t lB);

  /// Lorentzian metric, Armadillo vectorized, arma::Row interface
  template<typename F>
  F lorentzian(arma::Row<F> const& A, arma::Row<F> const& B);

  /// Minkowski metric, Armadillo vectorized, pointer interface
  ///  - Equal to the Manhattan distance with p=1
  ///  - Equal to the Euclidean Distance distance with p=2
  ///  - Also see the direct alignment function, which does the same without taking the root.
  ///    With exponent 0.5, 1, and 2, the direct alignment uses specialised cost function,
  ///    which may be faster, in particular for NN search.
  template<typename F>
  F minkowski(F const *A, size_t lA, F const *B, size_t lB, F p);

  /// Minkowski metric, Armadillo vectorized, arma::Row interface
  /// See the pointer interface above
  template<typename F>
  F minkowski(arma::Row<F> const& A, arma::Row<F> const& B, F p);

  /// Manhattan metric, Armadillo vectorized, pointer interface
  /// Special case for Minkowski with p=1
  template<typename F>
  F manhattan(F const *A, size_t lA, F const *B, size_t lB);

  /// Manhattan metric, Armadillo vectorized, arma::Row interface
  /// Special case for Minkowski with p=1
  template<typename F>
  F manhattan(arma::Row<F> const& A, arma::Row<F> const& B);


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Sliding
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /// SBD, Armadillo vectorized, pointer interface
  template<typename F>
  F sbd(F const *A, size_t lA, F const *B, size_t lB);

  /// SBD, Armadillo vectorized, arma::Row interface
  template<typename F>
  F sbd(arma::Row<F> const& A, arma::Row<F> const& B);


} // End of namespace tempo::distance::univariate
