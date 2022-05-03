#define CATCH_CONFIG_FAST_COMPILE

#include <mock/mockseries.hpp>
#include <catch.hpp>
#include <libtempo/distance/twe.hpp>


using namespace mock;
using namespace libtempo::distance::univariate;
constexpr size_t nbitems = 500;

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Reference
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

namespace reference {

  using namespace std;
  using namespace mock;
  using namespace libtempo::utils;

  /// Based on reference implementation by TWE author Pierre-François Marteau.
  /// * Added some checks.
  /// * Fix degree to 2 when evaluating the distance
  /// * Return TWE instead of modifying a pointed variable
  double twe_Marteau(const vector<double>& ta, const vector<double>& tb, double nu, double lambda) {
    ssize_t r = ta.size();
    ssize_t c = tb.size();

    const double deg = 2;
    double disti1, distj1, dist;
    long i, j;

    // Check lengths. Be explicit in the conditions.
    if (r==0 && c==0) { return 0; }
    if (r==0 && c!=0) { return PINF<double>; }
    if (r!=0 && c==0) { return PINF<double>; }

    // allocations
    double** D = (double**) calloc(r+1, sizeof(double*));
    double* Di1 = (double*) calloc(r+1, sizeof(double));
    double* Dj1 = (double*) calloc(c+1, sizeof(double));

    double dmin, htrans, dist0;

    for (i = 0; i<=r; i++) { D[i] = (double*) calloc(c+1, sizeof(double)); }

    // local costs initializations
    for (j = 1; j<=c; j++) {
      distj1 = 0;
      if (j>1) {
        distj1 += pow(fabs(tb[j-2]-tb[j-1]), deg);
      } else { distj1 += pow(fabs(tb[j-1]), deg); }

      Dj1[j] = distj1;
    }

    for (i = 1; i<=r; i++) {
      disti1 = 0;
      if (i>1) {
        disti1 += pow(fabs(ta[i-2]-ta[i-1]), deg);
      } else { disti1 += pow(fabs(ta[i-1]), deg); }

      Di1[i] = disti1;

      for (j = 1; j<=c; j++) {
        (dist) = 0;
        (dist) += pow(fabs(ta[i-1]-tb[j-1]), deg);
        if (i>1 && j>1) {
          (dist) += pow(fabs(ta[i-2]-tb[j-2]), deg);
        }

        D[i][j] = dist;
      }
    }// for i

    // border of the cost matrix initialization
    D[0][0] = 0;
    for (i = 1; i<=r; i++) { D[i][0] = PINF<double>; }
    for (j = 1; j<=c; j++) { D[0][j] = PINF<double>; }

    for (i = 1; i<=r; i++) {
      for (j = 1; j<=c; j++) {
        htrans = fabs((double) ((i-1)-(j-1)));
        if (j>1 && i>1) {
          htrans += fabs((double) ((i-2)-(j-2)));
        }
        dist0 = D[i-1][j-1]+D[i][j]+(nu)*htrans;
        dmin = dist0;
        if (i>1) {
          htrans = ((double) ((i-1)-(i-2)));
        } else { htrans = (double) 1; }
        (dist) = Di1[i]+D[i-1][j]+(lambda)+(nu)*htrans;
        if (dmin>(dist)) {
          dmin = (dist);
        }
        if (j>1) {
          htrans = ((double) ((j-1)-(j-2)));
        } else { htrans = (double) 1; }
        (dist) = Dj1[j]+D[i][j-1]+(lambda)+(nu)*htrans;
        if (dmin>(dist)) {
          dmin = (dist);
        }
        D[i][j] = dmin;
      }
    }

    dist = D[r][c];

    // freeing
    for (i = 0; i<=r; i++) { free(D[i]); }
    free(D);
    free(Di1);
    free(Dj1);

    return dist;
  }

  /// Our own TWE reference code.
  /// Warning: in the code, keep parenthesis: mimic how the costs are calculated, giving the exact same order of operations
  double twe_matrix(const vector<double>& series1, const vector<double>& series2, double nu, double lambda) {
    const size_t length1 = series1.size();
    const size_t length2 = series2.size();

    // Check lengths. Be explicit in the conditions.
    if (length1==0 && length2==0) { return 0; }
    if (length1==0 && length2!=0) { return PINF<double>; }
    if (length1!=0 && length2==0) { return PINF<double>; }

    const size_t maxLength = max(length1, length2);
    vector<std::vector<double>> matrix(maxLength, std::vector<double>(maxLength, PINF<double>));

    const double nu_lambda = nu+lambda;
    const double nu2 = 2*nu;

    // Initialization: first cell, first column and first row
    matrix[0][0] = sqdist(series1[0], series2[0]);
    for (size_t i = 1; i<length1; i++) { matrix[i][0] = matrix[i-1][0]+(sqdist(series1[i], series1[i-1])+nu_lambda); }
    for (size_t j = 1; j<length2; j++) { matrix[0][j] = matrix[0][j-1]+(sqdist(series2[j], series2[j-1])+nu_lambda); }

    // Main Loop
    for (size_t i = 1; i<length1; i++) {
      for (size_t j = 1; j<length2; j++) {
        // Top: over the lines
        double t = matrix[i-1][j]+(sqdist(series1[i], series1[i-1])+nu_lambda);
        // Diagonal
        double d = matrix[i-1][j-1]+(sqdist(series1[i], series2[j])+sqdist(series1[i-1], series2[j-1])+nu2*absdiff(i, j));
        // Previous: over the columns
        double p = matrix[i][j-1]+(sqdist(series2[j], series2[j-1])+nu_lambda);
        //
        matrix[i][j] = min(t, d, p);
      }
    }

    // Output
    return matrix[length1-1][length2-1];
  }

}

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
TEST_CASE("Univariate TWE Fixed length", "[twe][univariate]") {
  // Setup univariate with fixed length
  Mocker mocker;
  const auto& nus = mocker.twe_nus;
  const auto& lambdas = mocker.twe_lambdas;
  const auto fset = mocker.vec_randvec(nbitems);

  SECTION("TWE(s,s) == 0") {
    for (const auto& s: fset) {
      for (auto nu: nus) {
        for (auto la: lambdas) {
          const double twe_ref_v = reference::twe_Marteau(s, s, nu, la);
          REQUIRE(twe_ref_v==0);

          const double twe_ref_mat_v = reference::twe_matrix(s, s, nu, la);
          REQUIRE(twe_ref_mat_v==0);

          const double twe_v = twe(s, s, nu, la);
          REQUIRE(twe_v==0);
        }
      }
    }
  }

  SECTION("TWE(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i+1];

      for (auto nu: nus) {
        for (auto la: lambdas) {
          const double twe_ref_v = reference::twe_Marteau(s1, s2, nu, la);
          const double tempo_v = twe(s1, s2, nu, la);
          INFO("Not exact same operation orders. Requires approximate equality. " << nu << " " << la)
          REQUIRE(twe_ref_v==Approx(tempo_v));
          const double twe_ref_mat_v = reference::twe_matrix(s1, s2, nu, la);
          INFO("Ref Matrix code: same order of operation, exact same float")
          REQUIRE(twe_ref_mat_v==tempo_v);
        }
      }
    }
  }

  SECTION("NN1 TWE") {
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

        for (auto nu: nus) {
          for (auto la: lambdas) {
            // --- --- --- --- --- --- --- --- --- --- --- ---
            const double v_ref = reference::twe_Marteau(s1, s2, nu, la);
            if (v_ref<bsf_ref) {
              idx_ref = j;
              bsf_ref = v_ref;
            }

            // --- --- --- --- --- --- --- --- --- --- --- ---
            const auto v = twe<double>(s1, s2, nu, la);
            if (v<bsf) {
              idx = j;
              bsf = v;
            }
            REQUIRE(idx_ref==idx);

            // --- --- --- --- --- --- --- --- --- --- --- ---
            const auto v_tempo = twe<double>(s1, s2, nu, la, bsf_tempo);
            if (v_tempo<bsf_tempo) {
              idx_tempo = j;
              bsf_tempo = v_tempo;
            }
            REQUIRE(idx_ref==idx_tempo);

          }
        }
      }
    }// End query loop
  }// End section
}

TEST_CASE("Univariate TWE Variable length", "[twe][univariate]") {
  // Setup univariate dataset with varying length
  Mocker mocker;
  const auto& nus = mocker.twe_nus;
  const auto& lambdas = mocker.twe_lambdas;
  const auto fset = mocker.vec_rs_randvec(nbitems);

  SECTION("TWE(s,s) == 0") {
    for (const auto& s: fset) {
      for (auto nu: nus) {
        for (auto la: lambdas) {
          const double twe_ref_v = reference::twe_Marteau(s, s, nu, la);
          REQUIRE(twe_ref_v==0);

          const double twe_ref_mat_v = reference::twe_matrix(s, s, nu, la);
          REQUIRE(twe_ref_mat_v==0);

          const auto twe_v = twe<double>(s, s, nu, la);
          REQUIRE(twe_v==0);
        }
      }
    }
  }

  SECTION("TWE(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i+1];
      for (auto nu: nus) {
        for (auto la: lambdas) {
          const double twe_ref_v = reference::twe_Marteau(s1, s2, nu, la);
          const double tempo_v = twe(s1, s2, nu, la);
          INFO("Not exact same operation orders. Requires approximative equality.")
          REQUIRE(twe_ref_v==Approx(tempo_v));
          const double twe_ref_mat_v = reference::twe_matrix(s1, s2, nu, la);
          INFO("Ref Matrix code: same order of operation, exact same float")
          REQUIRE(twe_ref_mat_v==tempo_v);
        }
      }
    }
  }

  SECTION("NN1 TWE") {
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

        for (auto nu: nus) {
          for (auto la: lambdas) {

            // --- --- --- --- --- --- --- --- --- --- --- ---
            const double v_ref = reference::twe_Marteau(s1, s2, nu, la);
            if (v_ref<bsf_ref) {
              idx_ref = j;
              bsf_ref = v_ref;
            }
            // --- --- --- --- --- --- --- --- --- --- --- ---
            const auto v = twe<double>(s1, s2, nu, la);
            if (v<bsf) {
              idx = j;
              bsf = v;
            }
            REQUIRE(idx_ref==idx);
            // --- --- --- --- --- --- --- --- --- --- --- ---
            const auto v_tempo = twe<double>(s1, s2, nu, la, bsf_tempo);
            if (v_tempo<bsf_tempo) {
              idx_tempo = j;
              bsf_tempo = v_tempo;
            }
            REQUIRE(idx_ref==idx_tempo);
          }
        }
      }
    }// End query loop
  }// End section

}