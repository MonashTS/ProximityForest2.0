#define CATCH_CONFIG_FAST_COMPILE

#include <catch.hpp>
#include <mock/mockseries.hpp>

#include <libtempo/distance/adtw.hpp>
#include <libtempo/utils/utils.hpp>

using namespace mock;
using namespace libtempo::distance;
constexpr size_t nbitems = 500;
constexpr auto dist = univariate::ad2<double, std::vector<double>>;
constexpr double INF = libtempo::utils::PINF<double>;

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Reference
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
namespace reference {

  using namespace libtempo::utils;

  /// Naive DTW with a window. Reference code.
  double adtw_matrix(const vector<double>& series1, const vector<double>& series2, double penalty) {
    const long length1 = to_signed(series1.size());
    const long length2 = to_signed(series2.size());
    // Check lengths. Be explicit in the conditions
    if (length1==0 && length2==0) { return 0; }
    if (length1==0 && length2!=0) { return PINF<double>; }
    if (length1!=0 && length2==0) { return PINF<double>; }

    // Allocate the working space: full matrix + space for borders (first column / first line)
    size_t msize = max(length1, length2)+1;
    vector<std::vector<double>> matrix(msize, std::vector<double>(msize, PINF<double>));

    // Initialisation (all the matrix is initialised at +INF)
    matrix[0][0] = 0;

    // For each line
    // Note: series1 and series2 are 0-indexed while the matrix is 1-indexed (0 being the borders)
    //       hence, we have i-1 and j-1 when accessing series1 and series2
    for (long i = 1; i<=length1; i++) {
      auto series1_i = series1[i-1];
      for (long j = 1; j<=length2; j++) {
        double g = sqdist(series1_i, series2[j-1]);
        double prev = matrix[i][j-1]+g+penalty;
        double diag = matrix[i-1][j-1]+g;
        double top = matrix[i-1][j]+g+penalty;
        matrix[i][j] = min(prev, std::min(diag, top));
      }
    }

    return matrix[length1][length2];
  }
}

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
TEST_CASE("Univariate ADTW Fixed length", "[adtw][univariate]") {
  Catch::StringMaker<double>::precision = 18;

  // Setup univariate with fixed length
  Mocker mocker;
  const auto& penalties = mocker.adtw_penalties;
  const auto fset = mocker.vec_randvec(nbitems);

  SECTION("ADTW(s,s) == 0") {
    for (const auto& s: fset) {
      for (double p: penalties) {
        const double adtw_ref_v = reference::adtw_matrix(s, s, p);
        REQUIRE(adtw_ref_v==0);
        const double adtw_v = adtw(s.size(), s.size(), dist(s, s), p);
        REQUIRE(adtw_v==0);
      }
    }
  }

  SECTION("ADTW(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i+1];
      for (double p: penalties) {
        const double adtw_ref_v = reference::adtw_matrix(s1, s2, p);
        INFO("Exact same operation order. Expect exact floating point equality.")
        const auto adtw_tempo = adtw(s1.size(), s2.size(), dist(s1, s2), p);
        REQUIRE(adtw_ref_v==adtw_tempo);
      }
    }
  }

  SECTION("NN1 ADTW") {
    // Query loop
    for (size_t i = 0; i<nbitems; i += 3) {
      const auto& s1 = fset[i];
      // Ref Variables
      size_t idx_ref = 0;
      double bsf_ref = INF;
      // Base Variables
      size_t idx = 0;
      double bsf = lu::PINF<double>;
      // EAP Variables
      size_t idx_tempo = 0;
      double bsf_tempo = INF;

      // NN1 loop
      for (size_t j = 0; j<nbitems; j += 5) {
        // Skip self.
        if (i==j) { continue; }
        const auto& s2 = fset[j];
        // Create the univariate squared Euclidean distance for our adtw functions
        for (double p: penalties) {
          // --- --- --- --- --- --- --- --- --- --- --- ---
          const double v_ref = reference::adtw_matrix(s1, s2, p);
          if (v_ref<bsf_ref) {
            idx_ref = j;
            bsf_ref = v_ref;
          }

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v = adtw(s1.size(), s2.size(), dist(s1, s2), p);
          if (v<bsf) {
            idx = j;
            bsf = v;
          }

          REQUIRE(idx_ref==idx);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v_tempo = adtw(s1.size(), s2.size(), dist(s1, s2), p, bsf_tempo);
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

TEST_CASE("Univariate ADTW Variable length", "[adtw][univariate]") {
  // Setup univariate dataset with varying length
  Mocker mocker(1);
  const auto& penalties = mocker.adtw_penalties;
  const auto fset = mocker.vec_rs_randvec(nbitems);

  SECTION("ADTW(s,s) == 0") {
    for (const auto& s: fset) {
      for (double p: penalties) {
        const double adtw_ref_v = reference::adtw_matrix(s, s, p);
        REQUIRE(adtw_ref_v==0);
        const auto adtw_v = adtw(s.size(), s.size(), dist(s, s), p);
        REQUIRE(adtw_v==0);
      }
    }
  }

  SECTION("ADTW(s1, s2)") {
    for (size_t i = 28; i<nbitems-1; ++i) {
      for(size_t pi=44; pi<penalties.size(); ++pi){
        const auto p = penalties[pi];
        const auto& s1 = fset[i];
        const auto& s2 = fset[i+1];
        const double adtw_ref_v = reference::adtw_matrix(s1, s2, p);
        INFO("Exact same operation order. Expect exact floating point equality.")
        const auto adtw_tempo_v = adtw(s1.size(), s2.size(), dist(s1, s2), p);
        INFO(i << " " << pi << " length s1 = " << s1.size() << "  length s2 = " << s2.size())
        REQUIRE(adtw_ref_v==adtw_tempo_v);
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

        for (double p: penalties) {
          // --- --- --- --- --- --- --- --- --- --- --- ---
          const double v_ref = reference::adtw_matrix(s1, s2, p);
          if (v_ref<bsf_ref) {
            idx_ref = j;
            bsf_ref = v_ref;
          }

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v = adtw(s1.size(), s2.size(), dist(s1, s2), p);
          if (v<bsf) {
            idx = j;
            bsf = v;
          }

          REQUIRE(idx_ref==idx);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v_tempo = adtw(s1.size(), s2.size(), dist(s1, s2), p, bsf_tempo);
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