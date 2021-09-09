#define CATCH_CONFIG_FAST_COMPILE

#include <catch.hpp>
#include <libtempo/distance/norm.hpp>

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

  double norm(const vector<double>& a, const vector<double>& b, size_t dim) {
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
TEST_CASE("Multivariate Dependent NORM Fixed length", "[norm][multivariate]") {
  // Setup univariate with fixed length
  Mocker mocker;
  mocker._dim = ndim;

  const auto fset = mocker.vec_randvec(nbitems);

  SECTION("NORM(s,s) == 0") {
    for (const auto& s: fset) {
      const double norm_ref_v = reference::norm(s, s, ndim);
      REQUIRE(norm_ref_v==0);

      const auto norm_v = norm<double>(s, s, ndim);
      REQUIRE(norm_v==0);
    }
  }

  SECTION("NORM(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i+1];

      const double norm_ref_v = reference::norm(s1, s2, ndim);
      INFO("Exact same operation order. Expect exact floating point equality.")

      const auto norm_tempo_v = norm<double>(s1, s2, ndim);
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
        const double v_ref = reference::norm(s1, s2, ndim);
        if (v_ref<bsf_ref) {
          idx_ref = j;
          bsf_ref = v_ref;
        }

        // --- --- --- --- --- --- --- --- --- --- --- ---
        const auto v = norm<double>(s1, s2, ndim);
        if (v<bsf) {
          idx = j;
          bsf = v;
        }

        REQUIRE(idx_ref==idx);

        // --- --- --- --- --- --- --- --- --- --- --- ---
        const auto v_tempo = norm<double>(s1, s2, ndim, bsf_tempo);
        if (v_tempo<bsf_tempo) {
          idx_tempo = j;
          bsf_tempo = v_tempo;
        }

        REQUIRE(idx_ref==idx_tempo);
      }
    }// End query loop
  }// End section
}
