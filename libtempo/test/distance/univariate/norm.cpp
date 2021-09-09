#define CATCH_CONFIG_FAST_COMPILE

#include <catch.hpp>
#include <libtempo/distance/norm.hpp>

#include "../mock/mockseries.hpp"

using namespace mock;
using namespace libtempo::distance;
constexpr size_t nbitems = 500;

namespace reference {

  template<typename V>
  double norm(const V& s1, const V& s2) {
    if (s1.size()!=s2.size()) { return libtempo::utils::PINF<double>; }
    double cost = 0;
    for (size_t i = 0; i<s1.size(); ++i) {
      const auto d = s1[i]-s2[i];
      cost += d*d;
    }
    return cost;
  }

}

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
TEST_CASE("Univariate NORM Fixed length", "[norm][univariate]") {
  // Setup univariate with fixed length
  Mocker mocker;

  const auto fset = mocker.vec_randvec(nbitems);

  SECTION("NORM(s,s) == 0") {
    for (const auto& s: fset) {
      const double norm_ref_v = reference::norm(s, s);
      REQUIRE(norm_ref_v==0);

      const auto norm_v = norm<double>(s, s);
      REQUIRE(norm_v==0);
    }
  }

  SECTION("NORM(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i+1];

      const double norm_ref_v = reference::norm(s1, s2);
      INFO("Exact same operation order. Expect exact floating point equality.")

      const auto norm_tempo_v = norm<double>(s1, s2);
      REQUIRE(norm_ref_v==norm_tempo_v);
    }
  }

  SECTION("NN1 NORM") {
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
      size_t idx_tempo = 0;
      double bsf_tempo = lu::PINF<double>;

      // NN1 loop
      for (size_t j = 0; j<nbitems; j += 5) {
        // Skip self.
        if (i==j) { continue; }
        const auto& s2 = fset[j];

        // --- --- --- --- --- --- --- --- --- --- --- ---
        const double v_ref = reference::norm(s1, s2);
        if (v_ref<bsf_ref) {
          idx_ref = j;
          bsf_ref = v_ref;
        }

        // --- --- --- --- --- --- --- --- --- --- --- ---
        const auto v = norm<double>(s1, s2);
        if (v<bsf) {
          idx = j;
          bsf = v;
        }

        REQUIRE(idx_ref==idx);

        // --- --- --- --- --- --- --- --- --- --- --- ---
        const auto v_tempo = norm<double>(s1, s2, bsf_tempo);
        if (v_tempo<bsf_tempo) {
          idx_tempo = j;
          bsf_tempo = v_tempo;
        }

        REQUIRE(idx_ref==idx_tempo);
      }
    }// End query loop
  }// End section
}
