#define CATCH_CONFIG_FAST_COMPILE

#include <catch.hpp>
#include <libtempo/distance/direct.hpp>

#include "../mock/mockseries.hpp"

using namespace mock;
using namespace libtempo::distance;
constexpr size_t nbitems = 500;
constexpr size_t ndim = 3;

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Reference
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
namespace reference {

  using namespace libtempo::utils;
  using namespace std;

  /// Naive Univariate Squared euclidean distance. Reference code.
  template<typename V>
  double directa_uni(const V& s1, const V& s2) {
    if (s1.size()!=s2.size()) { return libtempo::utils::PINF<double>; }
    double cost = 0;
    for (size_t i = 0; i<s1.size(); ++i) {
      const auto d = s1[i]-s2[i];
      cost += d*d;
    }
    return cost;
  }

  /// Naive Univariate Squared euclidean distance. Reference code.
  double directa(const vector<double>& a, const vector<double>& b, size_t dim) {
    // Length of the series depends on the actual size of the data and the dimension
    const auto la = a.size()/dim;
    const auto lb = b.size()/dim;
    // Check lengths. Be explicit in the conditions.
    if (la==0 && lb==0) { return 0; }
    if (la==0 && lb!=0) { return PINF<double>; }
    if (la!=0 && lb==0) { return PINF<double>; }
    double cost = 0;
    for (size_t i = 0; i<la; ++i) { cost += sqedN(a, i, b, i, dim); }
    return cost;
  }
}


// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
TEST_CASE("Multivariate Dependent NORM Fixed length", "[directa][multivariate]") {
  // Setup univariate with fixed length
  Mocker mocker;
  mocker._dim = ndim;

  const auto fset = mocker.vec_randvec(nbitems);

  SECTION("NORM(s,s) == 0") {
    for (const auto& s: fset) {
      const double directa_ref_v = reference::directa(s, s, ndim);
      REQUIRE(directa_ref_v==0);

      const auto directa_v = directa<double>(s, s, ndim);
      REQUIRE(directa_v==0);
    }
  }

  SECTION("NORM(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i+1];

      // Check Uni
      {
        const double directa_ref_v = reference::directa(s1, s2, 1);
        const double directa_ref_uni_v = reference::directa_uni(s1, s2);
        const auto directa_tempo_v = directa<double>(s1, s2, 1);
        REQUIRE(directa_ref_v==directa_ref_uni_v);
        REQUIRE(directa_ref_v==directa_tempo_v);
      }

      // Check Multi
      {
        const double directa_ref_v = reference::directa(s1, s2, ndim);
        INFO("Exact same operation order. Expect exact floating point equality.")

        const auto directa_tempo_v = directa<double>(s1, s2, ndim);
        REQUIRE(directa_ref_v==directa_tempo_v);
      }
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
        const double v_ref = reference::directa(s1, s2, ndim);
        if (v_ref<bsf_ref) {
          idx_ref = j;
          bsf_ref = v_ref;
        }

        // --- --- --- --- --- --- --- --- --- --- --- ---
        const auto v = directa<double>(s1, s2, ndim);
        if (v<bsf) {
          idx = j;
          bsf = v;
        }

        REQUIRE(idx_ref==idx);

        // --- --- --- --- --- --- --- --- --- --- --- ---
        const auto v_tempo = directa<double>(s1, s2, ndim, bsf_tempo);
        if (v_tempo<bsf_tempo) {
          idx_tempo = j;
          bsf_tempo = v_tempo;
        }

        REQUIRE(idx_ref==idx_tempo);
      }
    }// End query loop
  }// End section
}
