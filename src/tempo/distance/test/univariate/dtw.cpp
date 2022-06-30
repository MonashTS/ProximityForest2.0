#define CATCH_CONFIG_FAST_COMPILE

#include <mock/mockseries.hpp>
#include <catch.hpp>
#include <tempo/distance/elastic/dtw.hpp>


// Configuration
constexpr size_t nbitems = 250;

using namespace mock;
using namespace tempo::distance;
using namespace tempo::utils;
// Using ad2 as the distance - same is used ("sqdist") in the ref code
constexpr auto dist = univariate::ad2<std::vector<double>>;

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Reference
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
namespace ref {

  /// Naive DTW with a window. Reference code.
  double cdtw_matrix(const vector<double>& series1, const vector<double>& series2, long w) {
    const long length1 = to_signed(series1.size());
    const long length2 = to_signed(series2.size());
    // Check lengths. Be explicit in the conditions
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
    for (long i = 1; i<=length1; i++) {
      auto series1_i = series1[i-1];
      long jStart = max<long>(1, i-w);
      long jStop = min<long>(i+w, length2);
      for (long j = jStart; j<=jStop; j++) {
        double prev = matrix[i][j-1];
        double diag = matrix[i-1][j-1];
        double top = matrix[i-1][j];
        matrix[i][j] = min(prev, std::min(diag, top))+sqdist(series1_i, series2[j-1]);
      }
    }

    return matrix[length1][length2];

  }
}

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
TEST_CASE("Univariate CDTW Fixed length", "[cdtw][univariate]") {
  // Setup univariate with fixed length
  Mocker mocker;
  const size_t l = 20;
  mocker._fixl = l;
  const auto& wratios = mocker.wratios;

  const auto fset = mocker.vec_randvec(nbitems);


  SECTION("CDTW(s,s) == 0") {
    for (const auto& s: fset) {
      for (double wr: wratios) {
        auto w = (size_t) (wr*mocker._fixl);
        const double cdtw_ref_v = ref::cdtw_matrix(s, s, w);
        REQUIRE(cdtw_ref_v==0);

        const auto cdtw_v = dtw(s.size(), s.size(), dist(s,s), w, PINF);
        REQUIRE(cdtw_v==0);

        const auto cdtw_wr = WR::dtw(s.size(), s.size(), dist(s,s), w, PINF);
        REQUIRE(cdtw_wr.cost==0);
        REQUIRE(cdtw_wr.window_validity==0);

      }
    }
  }

  SECTION("CDTW(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i+1];

      for (double wr: wratios) {
        const auto w = (size_t) (wr*l);

        const double cdtw_ref_v = ref::cdtw_matrix(s1, s2, w);
        INFO("Exact same operation order. Expect exact floating point equality.")

        const auto cdtw_tempo = dtw(s1.size(), s2.size(), dist(s1, s2), w, PINF);
        REQUIRE(cdtw_ref_v<=cdtw_tempo);

        const auto cdtw_tempo_wr = WR::dtw(s1.size(), s2.size(), dist(s1, s2), w, PINF);
        REQUIRE(cdtw_ref_v<=cdtw_tempo_wr.cost);
        REQUIRE(cdtw_tempo_wr.window_validity< NO_WINDOW);
      }
    }
  }

  SECTION("NN1 CDTW") {
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
      // EAP Variables
      size_t idx_tempo_wr = 0;
      double bsf_tempo_wr = PINF;

      // NN1 loop
      for (size_t j = 0; j<nbitems; j += 5) {
        // Skip self.
        if (i==j) { continue; }
        const auto& s2 = fset[j];
        // Create the univariate squared Euclidean distance for our cdtw functions
        for (double wr: wratios) {
          const auto w = (size_t) (wr*mocker._fixl);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const double v_ref = ref::cdtw_matrix(s1, s2, w);
          if (v_ref<bsf_ref) {
            idx_ref = j;
            bsf_ref = v_ref;
          }

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v = dtw(s1.size(), s2.size(), dist(s1, s2), w, PINF);
          if (v<bsf) {
            idx = j;
            bsf = v;
          }

          REQUIRE(idx_ref==idx);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v_tempo = dtw(s1.size(), s2.size(), dist(s1, s2), w, bsf_tempo);
          if (v_tempo<bsf_tempo) {
            idx_tempo = j;
            bsf_tempo = v_tempo;
          }

          REQUIRE(idx_ref==idx_tempo);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto wr_tempo = WR::dtw(s1.size(), s2.size(), dist(s1, s2), w, bsf_tempo_wr);
          if (wr_tempo.cost<bsf_tempo_wr) {
            idx_tempo_wr = j;
            bsf_tempo_wr = wr_tempo.cost;
          }

          REQUIRE(idx_ref==idx_tempo_wr);
        }
      }
    }// End query loop
  }// End section

}

TEST_CASE("Univariate CDTW Variable length", "[cdtw][univariate]") {
  // Setup univariate dataset with varying length
  Mocker mocker;
  const auto& wratios = mocker.wratios;

  const auto fset = mocker.vec_rs_randvec(nbitems);

  SECTION("CDTW(s,s) == 0") {
    for (const auto& s: fset) {
      for (double wr: wratios) {
        const auto w = (size_t) (wr*(s.size()));
        const double cdtw_ref_v = ref::cdtw_matrix(s, s, w);
        REQUIRE(cdtw_ref_v==0);

        const auto cdtw_v = dtw(s.size(), s.size(), dist(s, s), w, PINF);
        REQUIRE(cdtw_v==0);

        const auto cdtw_wr = WR::dtw(s.size(), s.size(), dist(s, s), w, PINF);
        REQUIRE(cdtw_wr.cost==0);
        REQUIRE(cdtw_wr.window_validity==0);
      }
    }
  }

  SECTION("CDTW(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      for (double wr: wratios) {
        const auto& s1 = fset[i];
        const auto& s2 = fset[i+1];
        const auto w = (size_t) (wr*(min(s1.size(), s2.size())));

        const double cdtw_ref_v = ref::cdtw_matrix(s1, s2, w);
        INFO("Exact same operation order. Expect exact floating point equality.")
        INFO("s1 length = " << s1.size())
        INFO("s2 length = " << s2.size())
        INFO("w = " << w)

        const auto cdtw_tempo_v = dtw(s1.size(), s2.size(), dist(s1, s2), w, PINF);
        REQUIRE(cdtw_ref_v==cdtw_tempo_v);

        const auto cdtw_tempo_wr = WR::dtw(s1.size(), s2.size(), dist(s1, s2), w, PINF);
        REQUIRE(cdtw_ref_v==cdtw_tempo_wr.cost);
      }
    }
  }

  SECTION("NN1 CDTW") {
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
      // EAP Variables
      size_t idx_tempo_wr = 0;
      double bsf_tempo_wr = PINF;

      // NN1 loop
      for (size_t j = 0; j<nbitems; j += 5) {
        // Skip self.
        if (i==j) { continue; }
        const auto& s2 = fset[j];
        // Create the univariate squared Euclidean distance for our cdtw functions

        for (double wr: wratios) {
          const auto w = (size_t) (wr*(min(s1.size(), s2.size())));

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const double v_ref = ref::cdtw_matrix(s1, s2, w);
          if (v_ref<bsf_ref) {
            idx_ref = j;
            bsf_ref = v_ref;
          }

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v = dtw(s1.size(), s2.size(), dist(s1, s2), w, PINF);
          if (v<bsf) {
            idx = j;
            bsf = v;
          }

          REQUIRE(idx_ref==idx);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v_tempo = dtw(s1.size(), s2.size(), dist(s1, s2), w, bsf_tempo);
          if (v_tempo<bsf_tempo) {
            idx_tempo = j;
            bsf_tempo = v_tempo;
          }

          REQUIRE(idx_ref==idx_tempo);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto wr_tempo = WR::dtw(s1.size(), s2.size(), dist(s1, s2), w, bsf_tempo_wr);
          if (wr_tempo.cost<bsf_tempo_wr) {
            idx_tempo_wr = j;
            bsf_tempo_wr = wr_tempo.cost;
          }

          REQUIRE(idx_ref==idx_tempo_wr);
        }
      }
    }// End query loop
  }// End section

}