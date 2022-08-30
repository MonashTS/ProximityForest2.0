#define CATCH_CONFIG_FAST_COMPILE

#include <mock/mockseries.hpp>
#include <catch.hpp>
#include <tempo/distance/elastic/dtw.hpp>

using namespace mock;
using namespace tempo::distance;
using namespace tempo::utils;
constexpr size_t nbitems = 500;
// Using ad2 as the distance - same is used ("sqdist") in the ref code
constexpr auto dist = univariate::ad2< std::vector<double>>;


// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Reference
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
namespace ref {

  /// Naive DTW without a window. Reference code.
  double dtw_matrix(const vector<double>& series1, const vector<double>& series2) {
    const auto length1 = series1.size();
    const auto length2 = series2.size();
    // Check lengths. Be explicit in the conditions.
    if (length1==0 && length2==0) { return 0; }
    if (length1==0 && length2!=0) { return PINF; }
    if (length1!=0 && length2==0) { return PINF; }

    // Allocate the working space: full matrix + space for borders (first column / first line)
    size_t msize = max(length1, length2)+1;
    vector<std::vector<double>> matrix(msize, std::vector<double>(msize, PINF));

    // Initialisation (all the matrix is initialised at +INF)
    matrix[0][0] = 0;

    // For each line
    // Note: series1 and series2 are 0-indexed while the matrix is 1-indexed (0 being the borders)
    //       hence, we have i-1 and j-1 when accessing series1 and series2
    for (size_t i = 1; i<=length1; i++) {
      auto series1_i = series1[i-1];
      for (size_t j = 1; j<=length2; j++) {
        double prev = matrix[i][j-1];
        double diag = matrix[i-1][j-1];
        double top = matrix[i-1][j];
        matrix[i][j] = min(prev, std::min(diag, top))+sqdist(series1_i, series2[j-1]);
      }
    }

    return matrix[length1][length2];
  }

}

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
TEST_CASE("Univariate DTW Fixed length", "[dtw][univariate]") {
  // Setup univariate with fixed length
  Mocker mocker;

  const auto fset = mocker.vec_randvec(nbitems);

  SECTION("DTW(s,s) == 0") {
    for (const auto& s: fset) {
      const double dtw_ref_v = ref::dtw_matrix(s, s);
      REQUIRE(dtw_ref_v==0);

      const auto dtw_v = dtw(s.size(), s.size(), dist(s, s), NO_WINDOW, PINF);
      REQUIRE(dtw_v==0);

      const auto dtw_wr = WR::dtw(s.size(), s.size(), dist(s, s), NO_WINDOW, PINF);
      REQUIRE(dtw_wr.cost==0);
      REQUIRE(dtw_wr.window_validity==0);
    }
  }

  SECTION("DTW(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i+1];

      const double dtw_ref_v = ref::dtw_matrix(s1, s2);
      INFO("Exact same operation order. Expect exact floating point equality.")

      const auto dtw_tempo_v = dtw(s1.size(), s2.size(), dist(s1, s2), NO_WINDOW, PINF);
      REQUIRE(dtw_ref_v==dtw_tempo_v);

      const auto dtw_tempo_wr = WR::dtw(s1.size(), s2.size(), dist(s1, s2), NO_WINDOW, PINF);
      REQUIRE(dtw_ref_v == dtw_tempo_wr.cost);
      REQUIRE(dtw_tempo_wr.window_validity<NO_WINDOW);
    }
  }

  SECTION("NN1 DTW") {
    // Query loop
    for (size_t i = 0; i<nbitems; i += 3) {
      const auto& s1 = fset[i];
      // Ref Variables
      size_t idx_ref = 0;
      double bsf_ref = PINF;
      // Base Variables
      size_t idx = 0;
      double bsf = PINF;
      // EAP Variables
      size_t idx_tempo = 0;
      double bsf_tempo = PINF;
      // EAP Variables WR
      size_t idx_tempo_wr = 0;
      double bsf_tempo_wr = PINF;

      // NN1 loop
      for (size_t j = 0; j<nbitems; j += 5) {
        // Skip self.
        if (i==j) { continue; }
        const auto& s2 = fset[j];

        // --- --- --- --- --- --- --- --- --- --- --- ---
        const double v_ref = ref::dtw_matrix(s1, s2);
        if (v_ref<bsf_ref) {
          idx_ref = j;
          bsf_ref = v_ref;
        }

        // --- --- --- --- --- --- --- --- --- --- --- ---
        const auto v = dtw(s1.size(), s2.size(), dist(s1, s2), NO_WINDOW, PINF);
        if (v<bsf) {
          idx = j;
          bsf = v;
        }

        REQUIRE(idx_ref==idx);

        // --- --- --- --- --- --- --- --- --- --- --- ---
        const auto v_tempo = dtw(s1.size(), s2.size(), dist(s1, s2), NO_WINDOW, bsf_tempo);
        if (v_tempo<bsf_tempo) {
          idx_tempo = j;
          bsf_tempo = v_tempo;
        }

        REQUIRE(idx_ref==idx_tempo);

        // --- --- --- --- --- --- --- --- --- --- --- ---
        const auto wr_tempo = WR::dtw(s1.size(), s2.size(), dist(s1, s2), NO_WINDOW, bsf_tempo_wr).cost;
        if (wr_tempo<bsf_tempo_wr) {
          idx_tempo_wr = j;
          bsf_tempo_wr = wr_tempo;
        }

        REQUIRE(idx_ref==idx_tempo_wr);
      }
    }// End query loop
  }// End section
}

TEST_CASE("Univariate DTW Variable length", "[dtw][univariate]") {
  // Setup univariate dataset with varying length
  Mocker mocker;

  const auto fset = mocker.vec_rs_randvec(nbitems);

  SECTION("DTW(s,s) == 0") {
    for (const auto& s: fset) {
      const double dtw_ref_v = ref::dtw_matrix(s, s);
      REQUIRE(dtw_ref_v==0);

      const auto dtw_v = dtw(s.size(), s.size(), dist(s, s), NO_WINDOW, PINF);
      REQUIRE(dtw_v==0);

      const auto dtw_wr = WR::dtw(s.size(), s.size(), dist(s, s), NO_WINDOW, PINF);
      REQUIRE(dtw_wr.cost==0);
      REQUIRE(dtw_wr.window_validity==0);
    }
  }

  SECTION("DTW(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i+1];

      const double dtw_ref_v = ref::dtw_matrix(s1, s2);
      INFO("Exact same operation order. Expect exact floating point equality.")

      const auto dtw_tempo_v = dtw(s1.size(), s2.size(), dist(s1, s2), NO_WINDOW, PINF);
      REQUIRE(dtw_ref_v==dtw_tempo_v);

      const auto dtw_tempo_wr = WR::dtw(s1.size(), s2.size(), dist(s1, s2), NO_WINDOW, PINF).cost;
      REQUIRE(dtw_ref_v==dtw_tempo_wr);
    }
  }

  SECTION("NN1 DTW") {
    // Query loop
    for (size_t i = 0; i<nbitems; i += 3) {
      const auto& s1 = fset[i];
      // Ref Variables
      size_t idx_ref = 0;
      double bsf_ref = PINF;
      // Base Variables
      size_t idx = 0;
      double bsf = PINF;
      // EAP Variables
      size_t idx_tempo = 0;
      double bsf_tempo = PINF;
      // EAP Variables WR
      size_t idx_tempo_wr = 0;
      double bsf_tempo_wr = PINF;

      // NN1 loop
      for (size_t j = 0; j<nbitems; j += 5) {
        // Skip self.
        if (i==j) { continue; }
        const auto& s2 = fset[j];

        // --- --- --- --- --- --- --- --- --- --- --- ---
        const double v_ref = ref::dtw_matrix(s1, s2);
        if (v_ref<bsf_ref) {
          idx_ref = j;
          bsf_ref = v_ref;
        }

        // --- --- --- --- --- --- --- --- --- --- --- ---
        const auto v = dtw(s1.size(), s2.size(), dist(s1, s2), NO_WINDOW, PINF);
        if (v<bsf) {
          idx = j;
          bsf = v;
        }

        REQUIRE(idx_ref==idx);

        // --- --- --- --- --- --- --- --- --- --- --- ---
        const auto v_tempo = dtw(s1.size(), s2.size(), dist(s1, s2), NO_WINDOW, bsf_tempo);
        if (v_tempo<bsf_tempo) {
          idx_tempo = j;
          bsf_tempo = v_tempo;
        }

        REQUIRE(idx_ref==idx_tempo);

        // --- --- --- --- --- --- --- --- --- --- --- ---
        const auto wr_tempo = WR::dtw(s1.size(), s2.size(), dist(s1, s2), NO_WINDOW, bsf_tempo_wr).cost;
        if (wr_tempo<bsf_tempo_wr) {
          idx_tempo_wr = j;
          bsf_tempo_wr = wr_tempo;
        }

        REQUIRE(idx_ref==idx_tempo_wr);
      }
    }// End query loop
  }// End section

}
