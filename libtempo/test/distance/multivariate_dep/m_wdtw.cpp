#define CATCH_CONFIG_FAST_COMPILE

#include <catch.hpp>
#include <libtempo/distance/wdtw.hpp>
#include <iostream>

#include "../mock/mockseries.hpp"

using namespace mock;
using namespace libtempo::distance;
constexpr size_t nbitems = 500;
constexpr size_t ndim = 3;
constexpr size_t nbweights = 5;

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Reference
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
namespace {

  using namespace libtempo::utils;
  using namespace std;

  double sqedN(const vector<double>& a, size_t astart, const vector<double>& b, size_t bstart, size_t dim) {
    double acc{0};
    const size_t aoffset = astart*dim;
    const size_t boffset = bstart*dim;
    for (size_t i{0}; i<dim; ++i) {
      double di = a[aoffset+i]-b[boffset+i];
      acc += di*di;
    }
    return acc;
  }

  double wdtw_matrix(const vector<double>& a, const vector<double>& b, size_t dim, std::vector<double> weights) {
    // Length of the series depends on the actual size of the data and the dimension
    const long la = (long) a.size()/(long) dim;
    const long lb = (long) b.size()/(long) dim;

    // Check lengths. Be explicit in the conditions.
    if (la==0 && lb==0) { return 0; }
    if (la==0 && lb!=0) { return PINF<double>; }
    if (la!=0 && lb==0) { return PINF<double>; }

    // Matrix
    vector<std::vector<double>> matrix(la, std::vector<double>(lb, 0));
    // First value
    matrix[0][0] = weights[0]*sqedN(a, 0, b, 0, dim);
    // First line
    for (long i = 1; i<lb; i++) {
      matrix[0][i] = matrix[0][i-1]+weights[i]*sqedN(a, 0, b, i, dim);
    }
    // First column
    for (long i = 1; i<la; i++) {
      matrix[i][0] = matrix[i-1][0]+weights[i]*sqedN(a, i, b, 0, dim);
    }
    // Matrix computation
    for (long i = 1; i<la; i++) {
      for (long j = 1; j<lb; j++) {
        const auto d = weights[abs(i-j)]*sqedN(a, i, b, j, dim);
        const auto v = min(matrix[i][j-1], std::min(matrix[i-1][j], matrix[i-1][j-1]))+d;
        matrix[i][j] = v;
      }
    }
    return matrix[la-1][lb-1];
  }

}

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
TEST_CASE("Multivariate Dependent WDTW Fixed length", "[wdtw][multivariate]") {
  // Setup univariate with fixed length
  Mocker mocker;
  const auto fset = mocker.vec_randvec(nbitems);
  // Random weight factors
  const auto weight_factors = mocker.randvec(nbweights, 0, 1);

  SECTION("DTW(s,s) == 0") {
    for (const auto& s: fset) {
      for (double g:weight_factors) {
        auto weights = generate_weights(g, mocker._fixl);

        const double dtw_ref_v = wdtw_matrix(s, s, ndim, weights);
        REQUIRE(dtw_ref_v==0);

        const auto dtw_v = wdtw<double>(s, s, ndim, weights);
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

        const double dtw_ref_v = wdtw_matrix(s1, s2, ndim, weights);
        INFO("Exact same operation order. Expect exact floating point equality.")

        const auto dtw_eap_v = wdtw<double>(s1, s2, ndim, weights);
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
          auto weights = generate_weights(g, mocker._fixl);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const double v_ref = wdtw_matrix(s1, s2, ndim, weights);
          if (v_ref<bsf_ref) {
            idx_ref = j;
            bsf_ref = v_ref;
          }

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v = wdtw<double>(s1, s2, ndim, weights);
          if (v<bsf) {
            idx = j;
            bsf = v;
          }

          REQUIRE(idx_ref==idx);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v_eap = wdtw<double>(s1, s2, ndim, weights);
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

TEST_CASE("Multivariate Dependent WDTW Variable length", "[wdtw][multivariate]") {
  // Setup univariate with fixed length
  Mocker mocker;
  const auto fset = mocker.vec_rs_randvec(nbitems);
  // Random weight factors
  const auto weight_factors = mocker.randvec(nbweights, 0, 1);

  SECTION("DTW(s,s) == 0") {
    for (const auto& s: fset) {
      for (double g:weight_factors) {
        auto weights = generate_weights(g, s.size());
        const double dtw_ref_v = wdtw_matrix(s, s, ndim, weights);
        REQUIRE(dtw_ref_v==0);

        const auto dtw_v = wdtw<double>(s, s, ndim, weights);
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

        const double dtw_ref_v = wdtw_matrix(s1, s2, ndim, weights);
        INFO("Exact same operation order. Expect exact floating point equality.")

        const auto dtw_eap_v = wdtw<double>(s1, s2, ndim, weights);
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

        for (double g:weight_factors) {
          auto weights = generate_weights(g, (min(s1.size(), s2.size())));

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const double v_ref = wdtw_matrix(s1, s2, ndim, weights);
          if (v_ref<bsf_ref) {
            idx_ref = j;
            bsf_ref = v_ref;
          }

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v = wdtw<double>(s1, s2, ndim, weights);
          if (v<bsf) {
            idx = j;
            bsf = v;
          }

          REQUIRE(idx_ref==idx);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v_eap = wdtw<double>(s1, s2, ndim, weights, bsf_eap);
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
