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

  /// Naive Univariate WDTW. Reference code.
  double wdtw_matrix_uni(const vector<double>& series1, const vector<double>& series2, const vector<double>& weights) {
    const long length1 = to_signed(series1.size());
    const long length2 = to_signed(series2.size());

    // Check lengths. Be explicit in the conditions.
    if (length1==0 && length2==0) { return 0; }
    if (length1==0 && length2!=0) { return PINF<double>; }
    if (length1!=0 && length2==0) { return PINF<double>; }
    // Matrix
    vector<std::vector<double>> matrix(length1, std::vector<double>(length2, 0));
    // First value
    matrix[0][0] = weights[0]*square_dist(series1[0], series2[0]);
    // First line
    for (long i = 1; i<length2; i++) {
      matrix[0][i] = matrix[0][i-1]+weights[i]*square_dist(series1[0], series2[i]);
    }
    // First column
    for (long i = 1; i<length1; i++) {
      matrix[i][0] = matrix[i-1][0]+weights[i]*square_dist(series1[i], series2[0]);
    }
    // Matrix computation
    for (long i = 1; i<length1; i++) {
      for (long j = 1; j<length2; j++) {
        const auto d = weights[abs(i-j)]*square_dist(series1[i], series2[j]);
        const auto v = min(matrix[i][j-1], std::min(matrix[i-1][j], matrix[i-1][j-1]))+d;
        matrix[i][j] = v;
      }
    }
    return matrix[length1-1][length2-1];
  }

  /// Naive Multivariate WDTW. Reference code.
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
TEST_CASE("Multivariate Dependent WWDTW Fixed length", "[wdtw][multivariate]") {
  // Setup univariate with fixed length
  Mocker mocker;
  const auto fset = mocker.vec_randvec(nbitems);
  // Random weight factors
  const auto weight_factors = mocker.randvec(nbweights, 0, 1);

  SECTION("WDTW(s,s) == 0") {
    for (const auto& s: fset) {
      for (double g: weight_factors) {
        auto weights = generate_weights(g, mocker._fixl);

        const double wdtw_ref_v = wdtw_matrix(s, s, ndim, weights);
        REQUIRE(wdtw_ref_v==0);

        const auto wdtw_v = wdtw<double>(s, s, ndim, weights);
        REQUIRE(wdtw_v==0);
      }
    }
  }

  SECTION("WDTW(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i+1];

      for (double g: weight_factors) {
        auto weights = generate_weights(g, mocker._fixl);

        // Check Uni
        {
          const double dtw_ref_v = wdtw_matrix(s1, s2, 1, weights);
          const double dtw_ref_uni_v = wdtw_matrix_uni(s1, s2, weights);
          const auto dtw_tempo_v = wdtw<double>(s1, s2, 1, weights);
          REQUIRE(dtw_ref_v==dtw_ref_uni_v);
          REQUIRE(dtw_ref_v==dtw_tempo_v);
        }

        // Check Multi
        {
          const double dtw_ref_v = wdtw_matrix(s1, s2, ndim, weights);
          INFO("Exact same operation order. Expect exact floating point equality.")

          const auto dtw_tempo_v = wdtw<double>(s1, s2, ndim, weights);
          REQUIRE(dtw_ref_v==dtw_tempo_v);
        }

      }
    }
  }

  SECTION("NN1 WDTW") {
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
        // Create the univariate squared Euclidean distance for our wdtw functions
        for (double g: weight_factors) {
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
          const auto v_tempo = wdtw<double>(s1, s2, ndim, weights, bsf_tempo);
          if (v_tempo<bsf_tempo) {
            idx_tempo = j;
            bsf_tempo = v_tempo;
          }

          REQUIRE(idx_ref==idx_tempo);
        }
      }
    }// End query loop
  }// End section
}

TEST_CASE("Multivariate Dependent WWDTW Variable length", "[wdtw][multivariate]") {
  // Setup univariate with fixed length
  Mocker mocker;
  const auto fset = mocker.vec_rs_randvec(nbitems);
  // Random weight factors
  const auto weight_factors = mocker.randvec(nbweights, 0, 1);

  SECTION("WDTW(s,s) == 0") {
    for (const auto& s: fset) {
      for (double g: weight_factors) {
        auto weights = generate_weights(g, s.size());
        const double wdtw_ref_v = wdtw_matrix(s, s, ndim, weights);
        REQUIRE(wdtw_ref_v==0);

        const auto wdtw_v = wdtw<double>(s, s, ndim, weights);
        REQUIRE(wdtw_v==0);
      }
    }
  }

  SECTION("WDTW(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      for (double g: weight_factors) {
        const auto& s1 = fset[i];
        const auto& s2 = fset[i+1];
        auto weights = generate_weights(g, (min(s1.size(), s2.size())));

        // Check Uni
        {
          const double dtw_ref_v = wdtw_matrix(s1, s2, 1, weights);
          const double dtw_ref_uni_v = wdtw_matrix_uni(s1, s2, weights);
          const auto dtw_tempo_v = wdtw<double>(s1, s2, 1, weights);
          REQUIRE(dtw_ref_v==dtw_ref_uni_v);
          REQUIRE(dtw_ref_v==dtw_tempo_v);
        }

        // Check Multi
        {
          const double dtw_ref_v = wdtw_matrix(s1, s2, ndim, weights);
          INFO("Exact same operation order. Expect exact floating point equality.")

          const auto dtw_tempo_v = wdtw<double>(s1, s2, ndim, weights);
          REQUIRE(dtw_ref_v==dtw_tempo_v);
        }
      }
    }
  }

  SECTION("NN1 WDTW") {
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

        for (double g: weight_factors) {
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
          const auto v_tempo = wdtw<double>(s1, s2, ndim, weights, bsf_tempo);
          if (v_tempo<bsf_tempo) {
            idx_tempo = j;
            bsf_tempo = v_tempo;
          }

          REQUIRE(idx_ref==idx_tempo);

        }
      }
    }// End query loop
  }// End section

}
