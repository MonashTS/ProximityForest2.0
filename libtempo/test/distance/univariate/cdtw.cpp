#define CATCH_CONFIG_FAST_COMPILE

#include <catch.hpp>
#include <libtempo/distance/cdtw.hpp>

#include "references/dtw/cdtw.hpp"
#include "../mock/mockseries.hpp"

using namespace mock;
using namespace libtempo::distance;
constexpr size_t nbitems = 500;

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
TEST_CASE("Univariate CDTW Fixed length", "[cdtw][univariate]") {
  // Setup univariate with fixed length
  Mocker mocker;
  const auto& wratios = mocker.wratios;

  const auto fset = mocker.vec_randvec(nbitems);

  SECTION("DTW(s,s) == 0") {
    for (const auto& s: fset) {
      for (double wr:wratios) {
        auto w = (size_t) (wr*mocker._fixl);
        const double dtw_ref_v = reference::cdtw_matrix(s, s, w);
        REQUIRE(dtw_ref_v==0);

        const auto dtw_v = cdtw<double>(s, s, w);
        REQUIRE(dtw_v==0);
      }
    }
  }

  SECTION("DTW(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i+1];

      for (double wr:wratios) {
        const auto w = (size_t) (wr*mocker._fixl);

        const double dtw_ref_v = reference::cdtw_matrix(s1, s2, w);
        INFO("Exact same operation order. Expect exact floating point equality.")

        const auto dtw_tempo = cdtw<double>(s1, s2, w);
        REQUIRE(dtw_ref_v==dtw_tempo);
      }
    }
  }

  SECTION("NN1 DTW") {
    // Query loop
    for (size_t i = 0; i<nbitems; i += 3) {
      const auto& s1 = fset[i];
      // Ref Variables
      size_t idx_ref = 0;
      double bsf_ref = lu::PINF<double>;
      // Base Variables
      size_t idx = 0;
      double bsf = lu::PINF<double>;
      // EAP Variables
      size_t idx_eap = 0;
      double bsf_eap = lu::PINF<double>;

      // NN1 loop
      for (size_t j = 0; j<nbitems; j += 5) {
        // Skip self.
        if (i==j) { continue; }
        const auto& s2 = fset[j];
        // Create the univariate squared Euclidean distance for our dtw functions
        for (double wr:wratios) {
          const auto w = (size_t) (wr*mocker._fixl);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const double v_ref = reference::cdtw_matrix(s1, s2, w);
          if (v_ref<bsf_ref) {
            idx_ref = j;
            bsf_ref = v_ref;
          }

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v = cdtw<double>(s1, s2, w);
          if (v<bsf) {
            idx = j;
            bsf = v;
          }

          REQUIRE(idx_ref==idx);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v_eap = cdtw<double>(s1, s2, w, bsf_eap);
          if (v_eap<bsf_eap) {
            idx_eap = j;
            bsf_eap = v_eap;
          }

          REQUIRE(idx_ref==idx_eap);
        }
      }
    }// End query loop
  }// End section

}

TEST_CASE("Univariate CDTW Variable length", "[cdtw][univariate]") {
  // Setup univariate dataset with varying length
  Mocker mocker;
  const auto& wratios = mocker.wratios;

  const auto fset = mocker.vec_rs_randvec(nbitems);

  SECTION("DTW(s,s) == 0") {
    for (const auto& s: fset) {
      for (double wr:wratios) {
        const auto w = (size_t) (wr*(s.size()));
        const double dtw_ref_v = reference::cdtw_matrix(s, s, w);
        REQUIRE(dtw_ref_v==0);

        const auto dtw_v = cdtw<double>(s, s, w);
        REQUIRE(dtw_v==0);
      }
    }
  }

  SECTION("DTW(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      for (double wr:wratios) {
        const auto& s1 = fset[i];
        const auto& s2 = fset[i+1];
        const auto w = (size_t) (wr*(min(s1.size(), s2.size())));

        const double dtw_ref_v = reference::cdtw_matrix(s1, s2, w);
        INFO("Exact same operation order. Expect exact floating point equality.")

        const auto dtw_eap_v = cdtw<double>(s1, s2, w);
        REQUIRE(dtw_ref_v==dtw_eap_v);
      }
    }
  }

  SECTION("NN1 DTW") {
    // Query loop
    for (size_t i = 0; i<nbitems; i += 3) {
      const auto& s1 = fset[i];
      // Ref Variables
      size_t idx_ref = 0;
      double bsf_ref = lu::PINF<double>;
      // Base Variables
      size_t idx = 0;
      double bsf = lu::PINF<double>;
      // EAP Variables
      size_t idx_eap = 0;
      double bsf_eap = lu::PINF<double>;

      // NN1 loop
      for (size_t j = 0; j<nbitems; j += 5) {
        // Skip self.
        if (i==j) { continue; }
        const auto& s2 = fset[j];
        // Create the univariate squared Euclidean distance for our dtw functions

        for (double wr:wratios) {
          const auto w = (size_t) (wr*(min(s1.size(), s2.size())));

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const double v_ref = reference::cdtw_matrix(s1, s2, w);
          if (v_ref<bsf_ref) {
            idx_ref = j;
            bsf_ref = v_ref;
          }

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v = cdtw<double>(s1, s2, w);
          if (v<bsf) {
            idx = j;
            bsf = v;
          }

          REQUIRE(idx_ref==idx);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v_eap = cdtw<double>(s1, s2, w, bsf_eap);
          if (v_eap<bsf_eap) {
            idx_eap = j;
            bsf_eap = v_eap;
          }

          REQUIRE(idx_ref==idx_eap);
        }
      }
    }// End query loop
  }// End section

}
