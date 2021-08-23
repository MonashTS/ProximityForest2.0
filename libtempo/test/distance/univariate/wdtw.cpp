#define CATCH_CONFIG_FAST_COMPILE

#include <catch.hpp>
#include <libtempo/distance/wdtw.hpp>

#include "references/dtw/wdtw.hpp"
#include "../mock/mockseries.hpp"

using namespace mock;
using namespace libtempo::distance;
constexpr size_t nbitems = 500;
constexpr size_t nbweights = 5;

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

TEST_CASE("Test weights generation", "[wdtw]") {
  // Random weight factors
  Mocker mocker;
  const auto weight_factors = mocker.randvec(nbweights, 0, 1);
  constexpr size_t maxlength = 50;
  constexpr double mc = ((double) maxlength)/2;
  for (double g: weight_factors) {
    auto weights = generate_weights(g, maxlength);
    for (size_t i = 0; i<maxlength; ++i) {
      auto lib = weights[i];
      auto ref = reference::mlwf((double)i, mc, g);
      REQUIRE(lib==ref);
    }
  }
}

TEST_CASE("Univariate WDTW Fixed length", "[wdtw][univariate]") {
  // Setup univariate with fixed length
  Mocker mocker;
  const auto fset = mocker.vec_randvec(nbitems);
  // Random weight factors
  const auto weight_factors = mocker.randvec(nbweights, 0, 1);

  SECTION("DTW(s,s) == 0") {
    for (const auto& s: fset) {
      for (double g:weight_factors) {
        auto weights = generate_weights(g, mocker._fixl);

        const double dtw_ref_v = reference::wdtw_matrix(s, s, weights);
        REQUIRE(dtw_ref_v==0);

        const auto dtw_v = wdtw<double>(s, s, weights);
        REQUIRE(dtw_v==0);
      }
    }
  }

  SECTION("DTW(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i+1];

      for (double g:weight_factors) {
        auto weights = generate_weights(g, mocker._fixl);

        const double dtw_ref_v = reference::wdtw_matrix(s1, s2, weights);
        INFO("Exact same operation order. Expect exact floating point equality.")

        const auto dtw_tempo = wdtw<double>(s1, s2, weights);
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
        for (double g:weight_factors) {
          auto weights = generate_weights(g, mocker._fixl);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const double v_ref = reference::wdtw_matrix(s1, s2, weights);
          if (v_ref<bsf_ref) {
            idx_ref = j;
            bsf_ref = v_ref;
          }

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v = wdtw<double>(s1, s2, weights);
          if (v<bsf) {
            idx = j;
            bsf = v;
          }

          REQUIRE(idx_ref==idx);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v_eap = wdtw<double>(s1, s2, weights, bsf_eap);
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

TEST_CASE("Univariate WDTW Variable length", "[wdtw][univariate]") {
  // Setup univariate with fixed length
  Mocker mocker;
  const auto fset = mocker.vec_rs_randvec(nbitems);
  // Random weight factors
  const auto weight_factors = mocker.randvec(nbweights, 0, 1);

  SECTION("DTW(s,s) == 0") {
    for (const auto& s: fset) {
      for (double g:weight_factors) {
        auto weights = generate_weights(g, s.size());
        const double dtw_ref_v = reference::wdtw_matrix(s, s, weights);
        REQUIRE(dtw_ref_v==0);

        const auto dtw_v = wdtw<double>(s, s, weights);
        REQUIRE(dtw_v==0);
      }
    }
  }

  SECTION("DTW(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
        for (double g:weight_factors) {
        const auto& s1 = fset[i];
        const auto& s2 = fset[i+1];
        auto weights = generate_weights(g, (min(s1.size(), s2.size())));

        const double dtw_ref_v = reference::wdtw_matrix(s1, s2, weights);
        INFO("Exact same operation order. Expect exact floating point equality.")

        const auto dtw_eap_v = wdtw<double>(s1, s2, weights);
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

        for (double g:weight_factors) {
          auto weights = generate_weights(g, (min(s1.size(), s2.size())));

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const double v_ref = reference::wdtw_matrix(s1, s2, weights);
          if (v_ref<bsf_ref) {
            idx_ref = j;
            bsf_ref = v_ref;
          }

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v = wdtw<double>(s1, s2, weights);
          if (v<bsf) {
            idx = j;
            bsf = v;
          }

          REQUIRE(idx_ref==idx);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v_eap = wdtw<double>(s1, s2, weights, bsf_eap);
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
