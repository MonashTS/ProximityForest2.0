#define CATCH_CONFIG_FAST_COMPILE

#include <catch.hpp>
#include <mock/mockseries.hpp>
#include <libtempo/distance/adtw.hpp>

using namespace mock;
using namespace libtempo::distance;
using namespace libtempo::distance::multivariate;
constexpr size_t nbitems = 1000;
constexpr size_t ndim = 3;
constexpr double INF = libtempo::utils::PINF<double>;

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Reference
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
namespace {

using namespace libtempo::utils;
using namespace std;

/// Naive univariate ADTW. Reference code.
double adtw_matrix_uni(const vector<double> &a, const vector<double> &b, double penalty) {
  const long la = to_signed(a.size());
  const long lb = to_signed(b.size());
  // Check lengths. Be explicit in the conditions
  if (la == 0 && lb == 0) { return 0; }
  if (la == 0 && lb != 0) { return PINF<double>; }
  if (la != 0 && lb == 0) { return PINF<double>; }

  // Allocate the working space: full matrix + space for borders (first column / first line)
  size_t msize = max(la, lb) + 1;
  vector<std::vector<double>> matrix(msize, std::vector<double>(msize, PINF<double>));

  // Initialisation (all the matrix is initialised at +INF)
  matrix[0][0] = 0;

  // For each line
  // Note: series1 and series2 are 0-indexed while the matrix is 1-indexed (0 being the borders)
  //       hence, we have i-1 and j-1 when accessing series1 and series2
  for (long i = 1; i <= la; i++) {
    for (long j = 1; j <= lb; j++) {
      double g = sqdist(a[i - 1], b[j - 1]);
      double prev = matrix[i][j - 1] + g + penalty;
      double diag = matrix[i - 1][j - 1] + g;
      double top = matrix[i - 1][j] + g + penalty;
      matrix[i][j] = min(prev, std::min(diag, top));
    }
  }

  return matrix[la][lb];
}

/// Naive multivariate ADTW. Reference code.
double adtw_matrix(const vector<double> &a, const vector<double> &b, size_t dim, double penalty) {
  const long la = (long) a.size() / (long) dim;
  const long lb = (long) b.size() / (long) dim;
  // Check lengths. Be explicit in the conditions
  if (la == 0 && lb == 0) { return 0; }
  if (la == 0 && lb != 0) { return PINF<double>; }
  if (la != 0 && lb == 0) { return PINF<double>; }

  // Allocate the working space: full matrix + space for borders (first column / first line)
  size_t msize = max(la, lb) + 1;
  vector<std::vector<double>> matrix(msize, std::vector<double>(msize, PINF<double>));

  // Initialisation (all the matrix is initialised at +INF)
  matrix[0][0] = 0;

  // For each line
  // Note: series1 and series2 are 0-indexed while the matrix is 1-indexed (0 being the borders)
  //       hence, we have i-1 and j-1 when accessing series1 and series2
  for (long i = 1; i <= la; i++) {
    for (long j = 1; j <= lb; j++) {
      double g = sqedN(a, i - 1, b, j - 1, dim);
      double prev = matrix[i][j - 1] + g + penalty;
      double diag = matrix[i - 1][j - 1] + g;
      double top = matrix[i - 1][j] + g + penalty;
      matrix[i][j] = min(prev, std::min(diag, top));
    }
  }

  return matrix[la][lb];
}

}

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
TEST_CASE("Multivariate Dependent ADTW Fixed length", "[adtw][multivariate]") {
  // Setup univariate with fixed length
  Mocker mocker;
  mocker._dim = ndim;
  const auto l = mocker._fixl;
  const auto l1 = mocker._fixl * ndim;
  const auto &penalties = mocker.adtw_penalties;

  const auto fset = mocker.vec_randvec(nbitems);

  SECTION("ADTW(s,s) == 0") {
    for (const auto &s: fset) {
      for (double p: penalties) {

        const double adtw_ref_v = adtw_matrix(s, s, ndim, p);
        REQUIRE(adtw_ref_v == 0);

        const double adtw_v = adtw<double>(l, l, p, ad2N<double>(s, s, ndim), INF);
        REQUIRE(adtw_v == 0);
      }
    }
  }

  SECTION("ADTW(s1, s2)") {
    for (size_t i = 0; i < nbitems - 1; ++i) {
      const auto &s1 = fset[i];
      const auto &s2 = fset[i + 1];

        for (double p: penalties) {

        // Check Uni
        {
          const double adtw_ref_v = adtw_matrix(s1, s2, 1, p);
          const double adtw_ref_uni_v = adtw_matrix_uni(s1, s2, p);
          const auto adtw_tempo_v = adtw(l1, l1, p, ad2N<double>(s1, s2, 1), INF);
          REQUIRE(adtw_ref_v == adtw_ref_uni_v);
          REQUIRE(adtw_ref_v == adtw_tempo_v);
        }

        // Check Multi
        {
          const double adtw_ref_v = adtw_matrix(s1, s2, ndim, p);
          INFO("Exact same operation order. Expect exact floating point equality.")

          const auto adtw_tempo_v = adtw(l, l, p, ad2N<double>(s1, s2, ndim), INF);
          REQUIRE(adtw_ref_v == adtw_tempo_v);
        }
      }
    }
  }

  SECTION("NN1 ADTW") {
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
        // Create the univariate squared Euclidean distance for our adtw functions

        for (double p:penalties) {
          // --- --- --- --- --- --- --- --- --- --- --- ---
          const double v_ref = adtw_matrix(s1, s2, ndim, p);
          if (v_ref<bsf_ref) {
            idx_ref = j;
            bsf_ref = v_ref;
          }

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v = adtw(l, l, p, ad2N<double>(s1, s2, ndim), INF);
          if (v<bsf) {
            idx = j;
            bsf = v;
          }

          REQUIRE(idx_ref==idx);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v_tempo = adtw(l, l, p, ad2N<double>(s1, s2, ndim), bsf_tempo);
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


TEST_CASE("Multivariate Dependent ADTW Variable length", "[adtw][multivariate]") {
  // Setup univariate dataset with varying length
  Mocker mocker;
  const auto &penalties = mocker.adtw_penalties;
  const auto fset = mocker.vec_rs_randvec(nbitems);

  SECTION("ADTW(s,s) == 0") {
    for (const auto &s: fset) {
      for (double p: penalties) {
        const double adtw_ref_v = adtw_matrix(s, s, ndim, p);
        REQUIRE(adtw_ref_v == 0);

        const auto adtw_v = adtw<double>(s.size() / ndim, s.size() / ndim, p, ad2N<double>(s, s, ndim), INF);
        REQUIRE(adtw_v == 0);
      }
    }
  }

  SECTION("ADTW(s1, s2)") {
    for (size_t i = 0; i < nbitems - 1; ++i) {
      for (double p: penalties) {
        const auto &s1 = fset[i];
        const auto &s2 = fset[i + 1];

        // Check Uni
        {
          const double adtw_ref_v = adtw_matrix(s1, s2, 1, p);
          const double adtw_ref_uni_v = adtw_matrix_uni(s1, s2, p);
          const auto adtw_tempo_v = adtw<double>(s1.size(), s2.size(), p, ad2N<double>(s1, s2, 1), INF);
          REQUIRE(adtw_ref_v == adtw_ref_uni_v);
          REQUIRE(adtw_ref_v == adtw_tempo_v);
        }

        // Check Multi
        {
          const double adtw_ref_v = adtw_matrix(s1, s2, ndim, p);
          INFO("Exact same operation order. Expect exact floating point equality.")

          const auto adtw_tempo_v = adtw<double>(s1.size() / ndim, s2.size() / ndim, p, ad2N<double>(s1, s2, ndim), INF);
          REQUIRE(adtw_ref_v == adtw_tempo_v);
        }
      }
    }
  }

  SECTION("NN1 ADTW") {
    // Query loop
    for (size_t i = 0; i < nbitems; i += 3) {
      const auto &s1 = fset[i];
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
      for (size_t j = 0; j < nbitems; j += 5) {
        // Skip self.
        if (i == j) { continue; }
        const auto &s2 = fset[j];

        for (double p: penalties) {

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const double v_ref = adtw_matrix(s1, s2, ndim, p);
          if (v_ref < bsf_ref) {
            idx_ref = j;
            bsf_ref = v_ref;
          }

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v = adtw<double>(s1.size() / ndim, s2.size() / ndim, p, ad2N<double>(s1, s2, ndim), INF);
          if (v < bsf) {
            idx = j;
            bsf = v;
          }

          REQUIRE(idx_ref == idx);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v_tempo = adtw<double>(s1.size() / ndim, s2.size() / ndim, p, ad2N<double>(s1, s2, ndim), bsf_tempo);
          if (v_tempo < bsf_tempo) {
            idx_tempo = j;
            bsf_tempo = v_tempo;
          }

          REQUIRE(idx_ref == idx_tempo);

        }
      }
    }// End query loop
  }// End section
}