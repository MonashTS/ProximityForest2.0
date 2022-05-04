#define CATCH_CONFIG_FAST_COMPILE

#include <mock/mockseries.hpp>
#include <catch.hpp>
#include <tempo/distance/msm.hpp>

using namespace mock;
using namespace tempo::distance::univariate;
constexpr size_t nbitems = 500;

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Reference
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
namespace reference {

  using namespace tempo::utils;

  /// Implementing cost from the original paper
  /// c is the minimal cost of an operation
  inline double get_cost(double xi, double xi1, double yi, double c) {
    if ((xi1<=xi && xi<=yi) || (xi1>=xi && xi>=yi)) {
      return c;
    } else {
      return c+std::min(std::fabs(xi-xi1), std::fabs(xi-yi));
    }
  }

  /// Naive MSM with a window. Reference code.
  double msm_matrix(const vector<double>& series1, const vector<double>& series2, double c) {
    const long length1 = to_signed(series1.size());
    const long length2 = to_signed(series2.size());

    // Check lengths. Be explicit in the conditions.
    if (length1==0 && length2==0) { return 0; }
    if (length1==0 && length2!=0) { return PINF<double>; }
    if (length1!=0 && length2==0) { return PINF<double>; }

    const long maxLength = max(length1, length2);
    vector<std::vector<double>> cost(maxLength, std::vector<double>(maxLength, PINF<double>));

    // Initialization
    cost[0][0] = abs(series1[0]-series2[0]);
    for (long i = 1; i<length1; i++) {
      cost[i][0] = cost[i-1][0]+get_cost(series1[i], series1[i-1], series2[0], c);
    }
    for (long i = 1; i<length2; i++) {
      cost[0][i] = cost[0][i-1]+get_cost(series2[i], series1[0], series2[i-1], c);
    }

    // Main Loop
    for (long i = 1; i<length1; i++) {
      for (long j = 1; j<length2; j++) {
        double d1, d2, d3;
        d1 = cost[i-1][j-1]+abs(series1[i]-series2[j]);                         // Diag
        d2 = cost[i-1][j]+get_cost(series1[i], series1[i-1], series2[j], c);     // Prev
        d3 = cost[i][j-1]+get_cost(series2[j], series1[i], series2[j-1], c);      // Top
        cost[i][j] = min(d1, std::min(d2, d3));
      }
    }

    // Output
    return cost[length1-1][length2-1];
  }

}

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
TEST_CASE("Univariate MSM Fixed length", "[msm][univariate]") {
  // Setup univariate with fixed length
  Mocker mocker;
  const auto& msm_costs = mocker.msm_costs;
  const auto fset = mocker.vec_randvec(nbitems);

  SECTION("MSM(s,s) == 0") {
    for (const auto& s: fset) {
      for (auto c: msm_costs) {
        const double msm_ref_v = reference::msm_matrix(s, s, c);
        REQUIRE(msm_ref_v==0);

        const auto msm_v = msm<double>(s, s, c);
        REQUIRE(msm_v==0);
      }
    }
  }

  SECTION("MSM(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i+1];

      for (auto c: msm_costs) {
        const double msm_ref_v = reference::msm_matrix(s1, s2, c);
        INFO("Exact same operation order. Expect exact floating point equality.")

        const auto msm_tempo = msm<double>(s1, s2, c);
        REQUIRE(msm_ref_v==msm_tempo);
      }
    }
  }

  SECTION("NN1 MSM") {
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
        // Create the univariate squared Euclidean distance for our msm functions
        for (auto c: msm_costs) {
          // --- --- --- --- --- --- --- --- --- --- --- ---
          const double v_ref = reference::msm_matrix(s1, s2, c);
          if (v_ref<bsf_ref) {
            idx_ref = j;
            bsf_ref = v_ref;
          }
          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v = msm<double>(s1, s2, c);
          if (v<bsf) {
            idx = j;
            bsf = v;
          }
          REQUIRE(idx_ref==idx);
          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v_tempo = msm<double>(s1, s2, c, bsf_tempo);
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

TEST_CASE("Univariate MSM Variable length", "[msm][univariate]") {
  // Setup univariate dataset with varying length
  Mocker mocker;
  const auto& msm_costs = mocker.msm_costs;
  const auto fset = mocker.vec_rs_randvec(nbitems);

  SECTION("MSM(s,s) == 0") {
    for (const auto& s: fset) {
      for (auto c: msm_costs) {
        const double msm_ref_v = reference::msm_matrix(s, s, c);
        REQUIRE(msm_ref_v==0);

        const auto msm_v = msm<double>(s, s, c);
        REQUIRE(msm_v==0);
      }
    }
  }

  SECTION("MSM(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i+1];
      for (auto c: msm_costs) {
        const double msm_ref_v = reference::msm_matrix(s1, s2, c);
        INFO("Exact same operation order. Expect exact floating point equality.")

        const auto msm_tempo_v = msm<double>(s1, s2, c, tempo::utils::QNAN<double>);
        REQUIRE(msm_ref_v==msm_tempo_v);
      }
    }
  }

  SECTION("NN1 MSM") {
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
        // Create the univariate squared Euclidean distance for our msm functions
        for (auto c: msm_costs) {
          // --- --- --- --- --- --- --- --- --- --- --- ---
          const double v_ref = reference::msm_matrix(s1, s2, c);
          if (v_ref<bsf_ref) {
            idx_ref = j;
            bsf_ref = v_ref;
          }
          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v = msm<double>(s1, s2, c);
          if (v<bsf) {
            idx = j;
            bsf = v;
          }
          REQUIRE(idx_ref==idx);
          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v_tempo = msm<double>(s1, s2, c, bsf_tempo);
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
