#define CATCH_CONFIG_FAST_COMPILE

#include <catch.hpp>
#include <mock/mockseries.hpp>

#include <libtempo/distance/cdtw.hpp>


using namespace mock;
using namespace libtempo::distance::univariate;
constexpr double INF = libtempo::utils::PINF<double>;
constexpr double QNAN = libtempo::utils::QNAN<double>;
constexpr size_t nbitems = 250;

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Reference
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
namespace reference {

  using namespace libtempo::utils;

  /// Naive DTW with a window. Reference code.
  double cdtw_matrix(const vector<double>& series1, const vector<double>& series2, long w) {
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
        const double cdtw_ref_v = reference::cdtw_matrix(s, s, w);
        REQUIRE(cdtw_ref_v==0);

        const auto cdtw_v = cdtw<double>(s, s, w);
        REQUIRE(cdtw_v==0);
      }
    }
  }

  SECTION("CDTW(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i+1];

      for (double wr: wratios) {
        const auto w = (size_t) (wr*l);

        const double cdtw_ref_v = reference::cdtw_matrix(s1, s2, w);
        INFO("Exact same operation order. Expect exact floating point equality.")

        const auto cdtw_tempo = cdtw<double>(s1, s2, w);
        REQUIRE(cdtw_ref_v<=cdtw_tempo);
      }
    }
  }

  SECTION("NN1 CDTW") {
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
        // Create the univariate squared Euclidean distance for our cdtw functions
        for (double wr: wratios) {
          const auto w = (size_t) (wr*mocker._fixl);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const double v_ref = reference::cdtw_matrix(s1, s2, w);
          if (v_ref<bsf_ref) {
            idx_ref = j;
            bsf_ref = v_ref;
          }

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v = cdtw<double>(s1, s2, w);
          if (v<bsf) {
            idx = j;
            bsf = v;
          }

          REQUIRE(idx_ref==idx);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v_tempo = cdtw<double>(s1, s2, w, bsf_tempo);
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

TEST_CASE("Univariate CDTW Variable length", "[cdtw][univariate]") {
  // Setup univariate dataset with varying length
  Mocker mocker;
  const auto& wratios = mocker.wratios;

  const auto fset = mocker.vec_rs_randvec(nbitems);

  SECTION("CDTW(s,s) == 0") {
    for (const auto& s: fset) {
      for (double wr: wratios) {
        const auto w = (size_t) (wr*(s.size()));
        const double cdtw_ref_v = reference::cdtw_matrix(s, s, w);
        REQUIRE(cdtw_ref_v==0);

        const auto cdtw_v = cdtw<double>(s, s, w);
        REQUIRE(cdtw_v==0);
      }
    }
  }

  SECTION("CDTW(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      for (double wr: wratios) {
        const auto& s1 = fset[i];
        const auto& s2 = fset[i+1];
        const auto w = (size_t) (wr*(min(s1.size(), s2.size())));

        const double cdtw_ref_v = reference::cdtw_matrix(s1, s2, w);
        INFO("Exact same operation order. Expect exact floating point equality.")

        const auto cdtw_tempo_v = cdtw<double>(s1, s2, w);
        REQUIRE(cdtw_ref_v==cdtw_tempo_v);
      }
    }
  }

  SECTION("NN1 CDTW") {
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
        // Create the univariate squared Euclidean distance for our cdtw functions

        for (double wr: wratios) {
          const auto w = (size_t) (wr*(min(s1.size(), s2.size())));

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const double v_ref = reference::cdtw_matrix(s1, s2, w);
          if (v_ref<bsf_ref) {
            idx_ref = j;
            bsf_ref = v_ref;
          }

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v = cdtw<double>(s1, s2, w);
          if (v<bsf) {
            idx = j;
            bsf = v;
          }

          REQUIRE(idx_ref==idx);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v_tempo = cdtw<double>(s1, s2, w, bsf_tempo);
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