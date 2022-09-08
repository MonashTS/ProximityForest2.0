#include "univariate.private.hpp"

namespace tempo::distance::univariate {

  // Implementation through template explicit instantiation

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Double implementation
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  using F = double;

  // --- --- --- Elastic distances --- --- ---

  template F adtw(F const *data1, size_t length1, F const *data2, size_t length2, F cfe, F penalty, F cutoff);

  template F dtw(F const *data1, size_t length1, F const *data2, size_t length2, F cfe, size_t window, F cutoff);

  template F wdtw(F const *data1, size_t length1, F const *data2, size_t length2, F cfe, F const *weights, F cutoff);
  template void wdtw_weights(F g, F *weights_array, size_t length, F wmax);
  template void wdtw_weights(F g, std::vector<F>& weights, size_t length, F wmax);
  template std::vector<F> wdtw_weights(F g, size_t length, F wmax);

  template F erp(F const *data1, size_t length1, F const *data2, size_t length2,
                 F cfe, F gap_value, size_t window, F cutoff);

  template F lcss(F const *data1, size_t length1, F const *data2, size_t length2, F epsilon, size_t window, F cutoff);

  template F msm(F const *data1, size_t length1, F const *data2, size_t length2, F cost, F cutoff);

  template F twe(F const *data1, size_t length1, F const *data2, size_t length2, F nu, F lambda, F cutoff);


  // --- --- --- DTW Lower bounds --- --- ---

  // --- --- --- --- --- --- Envelopes

  template void get_keogh_envelopes<F>(F const *series, size_t length, F *upper, F *lower, size_t w);

  template void get_keogh_envelopes<F>(F const *series, size_t length,
                                       std::vector<F>& upper, std::vector<F>& lower,
                                       size_t w);

  template void get_keogh_up_envelope(F const *series, size_t length, F *upper, size_t w);

  template void get_keogh_up_envelope(F const *series, size_t length, std::vector<F>& upper, size_t w);

  template void get_keogh_lo_envelope(F const *series, size_t length, F *lower, size_t w);

  template void get_keogh_lo_envelope(F const *series, size_t length, std::vector<F>& lower, size_t w);

  template void get_keogh_envelopes_Webb(
    F const *series, size_t length, F *upper, F *lower, F *lower_upper, F *upper_lower, size_t w
  );

  template void get_keogh_envelopes_Webb(F const *series, size_t length,
                                         std::vector<F>& upper, std::vector<F>& lower,
                                         std::vector<F>& lower_upper, std::vector<F>& upper_lower,
                                         size_t w);


  // --- --- --- --- --- --- Lower bounds

  template F lb_Keogh(F const *query, size_t query_length, F const *upper, F const *lower, F cfe, F cutoff);

  template F lb_Keogh(F const *query, size_t lquery, std::vector<F> const& upper, std::vector<F> const& lower,
                      F cfe, F cutoff);

  //

  template F lb_Keogh2j(
    F const *series1, size_t length1, F const *upper1, F const *lower1,
    F const *series2, size_t length2, F const *upper2, F const *lower2,
    F cfe, F cutoff
  );

  template F lb_Keogh2j(
    F const *series1, size_t length1, std::vector<F> const& upper1, std::vector<F> const& lower1,
    F const *series2, size_t length2, std::vector<F> const& upper2, std::vector<F> const& lower2,
    F cfe, F cutoff
  );

  //

  template F lb_Enhanced(const F *query, size_t lquery,
                         const F *candidate, size_t lcandidate, const F *candidate_upper, const F *candidate_lower,
                         F cfe, size_t v, size_t w, F cutoff);

  template F lb_Enhanced(const F *query, size_t lquery,
                         const F *candidate, size_t lcandidate,
                         std::vector<F> const& candidate_up, std::vector<F> const& candidate_lo,
                         F cfe, size_t v, size_t w, F cutoff);

  //

  template F lb_Enhanced2j(
    const F *series1, size_t length1, const F *upper1, const F *lower1,
    const F *series2, size_t length2, const F *upper2, const F *lower2,
    F cfe, size_t v, size_t w, F cutoff
  );

  template F lb_Enhanced2j(
    const F *series1, size_t length1, std::vector<F> const& upper1, std::vector<F> const& lower1,
    const F *series2, size_t length2, std::vector<F> const& upper2, std::vector<F> const& lower2,
    F cfe, size_t v, size_t w, F cutoff
  );

  //

  template F lb_Webb(
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

  template F lb_Webb(
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


  // --- --- --- Lockstep distances --- --- ---

  template F directa(F const *data1, size_t length1, F const *data2, size_t length2, F cfe, F cutoff);

  template F lorentzian(F const *A, size_t lA, F const *B, size_t lB);

  template F lorentzian(arma::Row<F> const& A, arma::Row<F> const& B);

  template F minkowski(F const *A, size_t lA, F const *B, size_t lB, F p);

  template F minkowski(arma::Row<F> const& A, arma::Row<F> const& B, F p);

  template F manhattan(F const *A, size_t lA, F const *B, size_t lB);

  template F manhattan(arma::Row<F> const& A, arma::Row<F> const& B);

} // End of namespace tempo::distance:univariate
