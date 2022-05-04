#define CATCH_CONFIG_FAST_COMPILE

#include <mock/mockseries.hpp>
#include <catch.hpp>
#include <tempo/distance/direct.hpp>

using namespace mock;
using namespace tempo::distance;
constexpr size_t nbitems = 500;
constexpr auto dist = univariate::ad2<double, std::vector<double>>;
constexpr double INF = tempo::utils::PINF<double>;

namespace reference {

  template<typename V>
  double directa(const V& s1, const V& s2) {
    if (s1.size()!=s2.size()) { return tempo::utils::PINF<double>; }
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
TEST_CASE("Univariate NORM Fixed length", "[directa][univariate]") {
  // Setup univariate with fixed length
  Mocker mocker;

  const auto fset = mocker.vec_randvec(nbitems);

  SECTION("NORM(s,s) == 0") {
    for (const auto& s: fset) {
      const double directa_ref_v = reference::directa(s, s);
      REQUIRE(directa_ref_v==0);

      const auto directa_v = directa(s.size(), s.size(), dist(s, s), INF);
      REQUIRE(directa_v==0);
    }
  }

  SECTION("NORM(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i+1];

      const double directa_ref_v = reference::directa(s1, s2);
      INFO("Exact same operation order. Expect exact floating point equality.")

      const auto directa_tempo_v = directa(s1.size(), s2.size(), dist(s1, s2), INF);
      REQUIRE(directa_ref_v==directa_tempo_v);
    }
  }

  SECTION("NN1 NORM") {
    // Query loop
    for (size_t i = 0; i<nbitems; i += 3) {
      const auto& s1 = fset[i];
      // Ref Variables
      size_t idx_ref = 0;
      double bsf_ref = INF;
      // Base Variables
      size_t idx = 0;
      double bsf = INF;
      // EAP Variables
      size_t idx_tempo = 0;
      double bsf_tempo = INF;

      // NN1 loop
      for (size_t j = 0; j<nbitems; j += 5) {
        // Skip self.
        if (i==j) { continue; }
        const auto& s2 = fset[j];

        // --- --- --- --- --- --- --- --- --- --- --- ---
        const double v_ref = reference::directa(s1, s2);
        if (v_ref<bsf_ref) {
          idx_ref = j;
          bsf_ref = v_ref;
        }

        // --- --- --- --- --- --- --- --- --- --- --- ---
        const auto v = directa(s1.size(), s2.size(), dist(s1, s2), INF);
        if (v<bsf) {
          idx = j;
          bsf = v;
        }

        REQUIRE(idx_ref==idx);

        // --- --- --- --- --- --- --- --- --- --- --- ---
        const auto v_tempo = directa(s1.size(), s2.size(), dist(s1, s2), bsf_tempo);
        if (v_tempo<bsf_tempo) {
          idx_tempo = j;
          bsf_tempo = v_tempo;
        }

        REQUIRE(idx_ref==idx_tempo);
      }
    }// End query loop
  }// End section
}
