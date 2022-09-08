#pragma once

#include "utils.hpp"
#include "cost_functions.hpp"
// --- --- --- Elastic distances --- --- ---
#include "core/elastic/adtw.hpp"
#include "core/elastic/dtw.hpp"
#include "core/elastic/wdtw.hpp"
#include "core/elastic/erp.hpp"
#include "core/elastic/lcss.hpp"
#include "core/elastic/msm.hpp"
#include "core/elastic/twe.hpp"
// --- --- --- DTW Lower Bound --- --- ---
#include "core/elastic/dtw_lb_keogh.hpp"
#include "core/elastic/dtw_lb_enhanced.hpp"
#include "core/elastic/dtw_lb_webb.hpp"
// --- ------ Lock step distances --- --- ---
#include "core/lockstep/direct.hpp"
#include "core/lockstep/lockstep.univariate.hpp"

#include <cstddef>
#include <vector>

namespace tempo::distance::univariate {

  namespace {
    // Hide shorthand in local anonymous namespace to avoid polluting the file including this one
    namespace tdc = tempo::distance::core;
    namespace tdcu = tempo::distance::core::univariate;
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Elastic distances
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  template<typename F>
  F adtw(
    F const *dat1, size_t len1,
    F const *dat2, size_t len2,
    F cfe,
    F penalty,
    F cutoff
  ) {
    if (cfe==1.0) {
      return tdc::adtw<F>(len1, len2, idx_ad1<F, F const *>(dat1, dat2), penalty, cutoff);
    } else if (cfe==2.0) {
      return tdc::adtw<F>(len1, len2, idx_ad2<F, F const *>(dat1, dat2), penalty, cutoff);
    } else if (cfe==0.5) {
      return tdc::adtw<F>(len1, len2, idx_ad_sqrt<F, F const *>(dat1, dat2), penalty, cutoff);
    } else {
      return tdc::adtw<F>(len1, len2, idx_ade<F, F const *>(cfe)(dat1, dat2), penalty, cutoff);
    }
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  template<typename F>
  F dtw(
    F const *const dat1, size_t len1,
    F const *const dat2, size_t len2,
    F cfe,
    size_t w,
    F cutoff
  ) {
    if (cfe==1.0) {
      return tdc::dtw<F>(len1, len2, idx_ad1<F, F const *>(dat1, dat2), w, cutoff);
    } else if (cfe==2.0) {
      return tdc::dtw<F>(len1, len2, idx_ad2<F, F const *>(dat1, dat2), w, cutoff);
    } else if (cfe==0.5) {
      return tdc::dtw<F>(len1, len2, idx_ad_sqrt<F, F const *>(dat1, dat2), w, cutoff);
    } else {
      return tdc::dtw<F>(len1, len2, idx_ade<F, F const *>(cfe)(dat1, dat2), w, cutoff);
    }
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  template<typename F>
  F wdtw(F const *dat1, size_t len1,
         F const *dat2, size_t len2,
         F cfe,
         F const *weights,
         F cutoff
  ) {
    if (cfe==1.0) {
      return tdc::wdtw<F>(len1, len2, idx_ad1<F, F const *>(dat1, dat2), weights, cutoff);
    } else if (cfe==2.0) {
      return tdc::wdtw<F>(len1, len2, idx_ad2<F, F const *>(dat1, dat2), weights, cutoff);
    } else if (cfe==0.5) {
      return tdc::wdtw<F>(len1, len2, idx_ad_sqrt<F, F const *>(dat1, dat2), weights, cutoff);
    } else {
      return tdc::wdtw<F>(len1, len2, idx_ade<F, F const *>(cfe)(dat1, dat2), weights, cutoff);
    }
  }

  template<typename F>
  void wdtw_weights(F g, F *weights_array, size_t length, F wmax) {
    tdc::wdtw_weights(g, weights_array, length, wmax);
  }

  template<typename F>
  void wdtw_weights(F g, std::vector<F>& weights, size_t length, F wmax) {
    weights.resize(length);
    wdtw_weights(g, weights.data(), length, wmax);
  }

  template<typename F>
  std::vector<F> wdtw_weights(F g, size_t length, F wmax) {
    std::vector<F> weights(length);
    wdtw_weights(g, weights.data(), length, wmax);
    return weights;
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  template<typename F>
  F erp(
    F const *const dat1, size_t len1,
    F const *const dat2, size_t len2,
    F cfe,
    F gv,
    size_t w,
    F cutoff
  ) {
    if (cfe==1.0) {
      constexpr auto gvf = tdcu::idx_gvad1<F, F const *>;
      return tdc::erp<F>(len1, len2, gvf(dat1, gv), gvf(dat2, gv), idx_ad1<F, F const *>(dat1, dat2), w, cutoff);
    } else if (cfe==2.0) {
      constexpr auto gvf = tdcu::idx_gvad2<F, F const *>;
      return tdc::erp<F>(len1, len2, gvf(dat1, gv), gvf(dat2, gv), idx_ad2<F, F const *>(dat1, dat2), w, cutoff);
    } else if (cfe==0.5) {
      constexpr auto gvf = tdcu::idx_gvad_sqrt<F, F const *>;
      return tdc::erp<F>(len1, len2, gvf(dat1, gv), gvf(dat2, gv), idx_ad_sqrt<F, F const *>(dat1, dat2), w, cutoff);
    } else {
      auto gve = tdcu::idx_gvade<F, F const *>(cfe);
      return tdc::erp<F>(len1, len2, gve(dat1, gv), gve(dat2, gv), idx_ade<F, F const *>(cfe)(dat1, dat2), w, cutoff);
    }
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  template<typename F>
  F lcss(
    F const *const dat1, size_t len1,
    F const *const dat2, size_t len2,
    F e,
    size_t w,
    F cutoff
  ) {
    return tdc::lcss<F>(len1, len2, tdcu::idx_simdiff<F, F const *>(e)(dat1, dat2), w, cutoff);
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  template<typename F>
  F msm(F const *data1, size_t length1,
        F const *data2, size_t length2,
        F cost,
        F cutoff
  ) {
    constexpr auto cfli = tdcu::idx_msm_lines<F, F const *>;
    constexpr auto cfco = tdcu::idx_msm_cols<F, F const *>;
    constexpr auto cfdi = tdcu::idx_msm_diag<F, F const *>;
    return tdc::msm<F>(length1, length2, cfli(data1, data2, cost), cfco(data1, data2, cost), cfdi(data1, data2), cutoff
    );
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  template<typename F>
  F twe(F const *data1, size_t length1,
        F const *data2, size_t length2,
        F nu,
        F lambda,
        F cutoff
  ) {
    constexpr auto cfwarp = tdcu::idx_twe_warp<F, F const *>;
    constexpr auto cfmatch = tdcu::idx_twe_match<F, F const *>;
    return tdc::twe<F>(
      length1, length2, cfwarp(data1, nu, lambda), cfwarp(data2, nu, lambda), cfmatch(data1, data2, nu), cutoff
    );
  }



  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // DTW Lower Bound
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  template<typename F>
  void get_keogh_envelopes(F const *series, size_t length, F *upper, F *lower, size_t w) {
    tdcu::get_keogh_envelopes(series, length, upper, lower, w);
  }

  template<typename F>
  void get_keogh_envelopes(F const *series, size_t length, std::vector<F>& upper, std::vector<F>& lower, size_t w) {
    upper.resize(length);
    lower.resize(length);
    get_keogh_envelopes(series, length, upper.data(), lower.data(), w);
  }

  template<typename F>
  void get_keogh_up_envelope(F const *series, size_t length, F *upper, size_t w) {
    tdcu::get_keogh_up_envelope(series, length, upper, w);
  }

  template<typename F>
  void get_keogh_up_envelope(F const *series, size_t length, std::vector<F>& upper, size_t w) {
    upper.resize(length);
    get_keogh_up_envelope(series, length, upper.data(), w);
  }

  template<typename F>
  void get_keogh_lo_envelope(F const *series, size_t length, F *lower, size_t w) {
    tdcu::get_keogh_lo_envelope(series, length, lower, w);
  }

  template<typename F>
  void get_keogh_lo_envelope(F const *series, size_t length, std::vector<F>& upper, size_t w) {
    upper.resize(length);
    get_keogh_lo_envelope(series, length, upper.data(), w);
  }

  template<typename F>
  void get_keogh_envelopes_Webb(
    F const *series, size_t length, F *upper, F *lower, F *lower_upper, F *upper_lower, size_t w
  ) {
    tdcu::get_keogh_envelopes_Webb(series, length, upper, lower, lower_upper, upper_lower, w);
  }

  template<typename F>
  void get_keogh_envelopes_Webb(
    F const *series, size_t length, std::vector<F>& upper, std::vector<F>& lower,
    std::vector<F>& lower_upper, std::vector<F>& upper_lower,
    size_t w
  ) {
    get_keogh_envelopes(series, length, upper, lower, w);
    get_keogh_lo_envelope(upper.data(), length, lower_upper, w);
    get_keogh_up_envelope(lower.data(), length, upper_lower, w);
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  template<typename F>
  F lb_Keogh(F const *query, size_t query_length, F const *upper, F const *lower, F cfe, F cutoff) {
    if (cfe==1.0) {
      constexpr utils::CFun<F> auto cf = ad1<F>;
      return tdcu::lb_Keogh(query, query_length, upper, lower, cf, cutoff);
    } else if (cfe==2.0) {
      constexpr utils::CFun<F> auto cf = ad2<F>;
      return tdcu::lb_Keogh(query, query_length, upper, lower, cf, cutoff);
    } else if (cfe==0.5) {
      constexpr utils::CFun<F> auto cf = ad_sqrt<F>;
      return tdcu::lb_Keogh(query, query_length, upper, lower, cf, cutoff);
    } else {
      utils::CFun<F> auto cf = ade<F>(cfe);
      return tdcu::lb_Keogh(query, query_length, upper, lower, cf, cutoff);
    }
  }

  template<typename F>
  F lb_Keogh(F const *query, size_t lquery, std::vector<F> const& upper, std::vector<F> const& lower, F cfe, F cutoff) {
    return lb_Keogh(query, lquery, upper.data(), lower.data(), cfe, cutoff);
  }

  //

  template<typename F>
  F lb_Keogh2j(
    F const *series1, size_t length1, F const *upper1, F const *lower1,
    F const *series2, size_t length2, F const *upper2, F const *lower2,
    F cfe, F cutoff
  ) {
    if (cfe==1.0) {
      constexpr utils::CFun<F> auto cf = ad1<F>;
      return tdcu::lb_Keogh2j(series1, length1, upper1, lower1, series2, length2, upper2, lower2, cf, cutoff);
    } else if (cfe==2.0) {
      constexpr utils::CFun<F> auto cf = ad2<F>;
      return tdcu::lb_Keogh2j(series1, length1, upper1, lower1, series2, length2, upper2, lower2, cf, cutoff);
    } else if (cfe==0.5) {
      constexpr utils::CFun<F> auto cf = ad_sqrt<F>;
      return tdcu::lb_Keogh2j(series1, length1, upper1, lower1, series2, length2, upper2, lower2, cf, cutoff);
    } else {
      utils::CFun<F> auto cf = ade<F>(cfe);
      return tdcu::lb_Keogh2j(series1, length1, upper1, lower1, series2, length2, upper2, lower2, cf, cutoff);
    }
  }

  template<typename F>
  F lb_Keogh2j(
    F const *series1, size_t length1, std::vector<F> const& upper1, std::vector<F> const& lower1,
    F const *series2, size_t length2, std::vector<F> const& upper2, std::vector<F> const& lower2,
    F cfe, F cutoff
  ) {
    return lb_Keogh2j(
      series1, length1, upper1.data(), lower1.data(),
      series2, length2, upper2.data(), lower2.data(),
      cfe, cutoff
    );
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  template<typename F>
  F lb_Enhanced(const F *query, size_t lquery,
                const F *candidate, size_t lcandidate,
                const F *candidate_up, const F *candidate_lo,
                F cfe, size_t v, size_t w, F cutoff
  ) {
    if (cfe==1.0) {
      constexpr utils::CFun<F> auto cf = ad1<F>;
      return tdcu::lb_Enhanced(query, lquery, candidate, lcandidate, candidate_up, candidate_lo, cf, v, w, cutoff);
    } else if (cfe==2.0) {
      constexpr utils::CFun<F> auto cf = ad2<F>;
      return tdcu::lb_Enhanced(query, lquery, candidate, lcandidate, candidate_up, candidate_lo, cf, v, w, cutoff);
    } else if (cfe==0.5) {
      constexpr utils::CFun<F> auto cf = ad_sqrt<F>;
      return tdcu::lb_Enhanced(query, lquery, candidate, lcandidate, candidate_up, candidate_lo, cf, v, w, cutoff);
    } else {
      utils::CFun<F> auto cf = ade<F>(cfe);
      return tdcu::lb_Enhanced(query, lquery, candidate, lcandidate, candidate_up, candidate_lo, cf, v, w, cutoff);
    }
  }

  template<typename F>
  F lb_Enhanced(const F *query, size_t lquery,
                const F *candidate, size_t lcandidate,
                std::vector<F> const& candidate_up, std::vector<F> const& candidate_lo,
                F cfe, size_t v, size_t w, F cutoff
  ) {
    return
      lb_Enhanced(query, lquery, candidate, lcandidate, candidate_up.data(), candidate_lo.data(), cfe, v, w, cutoff);
  }

  //

  template<typename F>
  F lb_Enhanced2j(
    const F *series1, size_t length1, const F *upper1, const F *lower1,
    const F *series2, size_t length2, const F *upper2, const F *lower2,
    F cfe, size_t v, size_t w, F cutoff
  ) {
    if (cfe==1.0) {
      constexpr utils::CFun<F> auto cf = ad1<F>;
      return tdcu::lb_Enhanced2j(series1, length1, upper1, lower1, series2, length2, upper2, lower2, cf, v, w, cutoff);
    } else if (cfe==2.0) {
      constexpr utils::CFun<F> auto cf = ad2<F>;
      return tdcu::lb_Enhanced2j(series1, length1, upper1, lower1, series2, length2, upper2, lower2, cf, v, w, cutoff);
    } else if (cfe==0.5) {
      constexpr utils::CFun<F> auto cf = ad_sqrt<F>;
      return tdcu::lb_Enhanced2j(series1, length1, upper1, lower1, series2, length2, upper2, lower2, cf, v, w, cutoff);
    } else {
      utils::CFun<F> auto cf = ade<F>(cfe);
      return tdcu::lb_Enhanced2j(series1, length1, upper1, lower1, series2, length2, upper2, lower2, cf, v, w, cutoff);
    }
  }

  template<typename F>
  F lb_Enhanced2j(
    const F *series1, size_t length1, std::vector<F> const& upper1, std::vector<F> const& lower1,
    const F *series2, size_t length2, std::vector<F> const& upper2, std::vector<F> const& lower2,
    F cfe, size_t v, size_t w, F cutoff
  ) {
    return lb_Enhanced2j(
      series1, length1, upper1.data(), lower1.data(),
      series2, length2, upper2.data(), lower2.data(),
      cfe, v, w, cutoff
    );
  }


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---


  template<typename F>
  F lb_Webb(
    // Series A
    F const *a, size_t a_len,
    F const *a_up, F const *a_lo,
    F const *a_lo_up, F const *a_up_lo,
    // Series B
    F const *b, size_t b_len,
    F const *b_up, F const *b_lo,
    F const *b_lo_up, F const *b_up_lo,
    // Cost function
    F cfe,
    // Others
    size_t w, F cutoff //cutoff
  ) {
    if (cfe==1.0) {
      constexpr utils::CFun<F> auto cf = ad1<F>;
      return
        tdcu::lb_Webb(a, a_len, a_up, a_lo, a_lo_up, a_up_lo, b, b_len, b_up, b_lo, b_lo_up, b_up_lo, cf, w, cutoff);
    } else if (cfe==2.0) {
      constexpr utils::CFun<F> auto cf = ad2<F>;
      return
        tdcu::lb_Webb(a, a_len, a_up, a_lo, a_lo_up, a_up_lo, b, b_len, b_up, b_lo, b_lo_up, b_up_lo, cf, w, cutoff);
    } else if (cfe==0.5) {
      constexpr utils::CFun<F> auto cf = ad_sqrt<F>;
      return
        tdcu::lb_Webb(a, a_len, a_up, a_lo, a_lo_up, a_up_lo, b, b_len, b_up, b_lo, b_lo_up, b_up_lo, cf, w, cutoff);
    } else {
      utils::CFun<F> auto cf = ade<F>(cfe);
      return
        tdcu::lb_Webb(a, a_len, a_up, a_lo, a_lo_up, a_up_lo, b, b_len, b_up, b_lo, b_lo_up, b_up_lo, cf, w, cutoff);
    }
  }

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
    // Cost function
    F cfe,
    // Others
    size_t w, F cutoff
  ) {
    return lb_Webb(
      a, a_len, a_up.data(), a_lo.data(), a_lo_up.data(), a_up_lo.data(),
      b, b_len, b_up.data(), b_lo.data(), b_lo_up.data(), b_up_lo.data(),
      cfe, w, cutoff
    );
  }



  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Lockstep distances
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /// Direct alignment with cost function cfe, and early abandoning cutoff.
  template<typename F>
  F directa(F const *dat1, size_t len1, F const *dat2, size_t len2, F cfe, F cutoff) {
    if (cfe==1.0) {
      return tdc::directa<F>(len1, len2, idx_ad1<F, F const *>(dat1, dat2), cutoff);
    } else if (cfe==2.0) {
      return tdc::directa<F>(len1, len2, idx_ad2<F, F const *>(dat1, dat2), cutoff);
    } else if (cfe==0.5) {
      return tdc::directa<F>(len1, len2, idx_ad_sqrt<F, F const *>(dat1, dat2), cutoff);
    } else {
      return tdc::directa<F>(len1, len2, idx_ade<F, F const *>(cfe)(dat1, dat2), cutoff);
    }
  }

  template<typename F>
  F lorentzian(arma::Row<F> const& A, arma::Row<F> const& B) {

    return arma::sum(arma::log(1 + arma::abs(A - B)));
  }

  template<typename F>
  F lorentzian(F const *A, size_t lA, F const *B, size_t lB) {
    const arma::Row<F> ra(const_cast<F *>(A), lA, false, true);
    const arma::Row<F> rb(const_cast<F *>(B), lB, false, true);
    return lorentzian<F>(ra, rb);
  }

  template<typename F>
  F minkowski(arma::Row<F> const& A, arma::Row<F> const& B, F p) {
    return std::pow(arma::sum(arma::pow(arma::abs(A - B), p)), (F)1.0/p);
  }

  template<typename F>
  F minkowski(F const *A, size_t lA, F const *B, size_t lB, F p) {
    const arma::Row<F> ra(const_cast<F *>(A), lA, false, true);
    const arma::Row<F> rb(const_cast<F *>(B), lB, false, true);
    return minkowski<F>(ra, rb, p);
  }

  template<typename F>
  F manhattan(arma::Row<F> const& A, arma::Row<F> const& B) {
    return arma::sum(arma::abs(A - B));
  }

  template<typename F>
  F manhattan(F const *A, size_t lA, F const *B, size_t lB) {
    const arma::Row<F> ra(const_cast<F *>(A), lA, false, true);
    const arma::Row<F> rb(const_cast<F *>(B), lB, false, true);
    return manhattan<F>(ra, rb);
  }

} // End of namespace tempo::distance::univariate
