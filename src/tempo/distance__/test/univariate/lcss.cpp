#define CATCH_CONFIG_FAST_COMPILE

#include <mock/mockseries.hpp>
#include <catch.hpp>
#include <tempo/distance/elastic/lcss.hpp>

using namespace mock;
using namespace tempo::distance;
using namespace tempo::utils;
constexpr size_t nbitems = 500;
// LCSS by default is using |a-b|, i.e. ad1
constexpr auto dist = univariate::ad1<std::vector<double>>;

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Reference
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
namespace ref {

  /// Naive DTW with a window. Reference code.
  double lcss_matrix(const vector<double>& series1, const vector<double>& series2, long w, double epsilon) {
    constexpr double PINF = tempo::utils::PINF;
    size_t length1 = series1.size();
    size_t length2 = series2.size();

    // Check lengths. Be explicit in the conditions
    if (length1==0 && length2==0) { return 0; }
    if (length1==0 && length2!=0) { return PINF; }
    if (length1!=0 && length2==0) { return PINF; }

    // Allocate the working space: full matrix + space for borders (first column / first line)
    size_t maxLength = max<size_t>(length1, length2);
    size_t minLength = min<size_t>(length1, length2);
    vector<std::vector<int>> matrix(maxLength+1, std::vector<int>(maxLength+1, 0));

    // Marker for final point
    matrix[length1][length2] = -1;

    // LCSS
    // Note: series1 and series2 are 0-indexed while the matrix is 1-indexed (0 being the borders)
    //       hencw, ee have i-1 and j-1 when accessing series1 and series2
    for (long i = 1; i<=(long) length1; ++i) {
      auto series1_i = series1[i-1];
      long jStart = max<long>(1, i-w);
      long jStop = min<long>(i+w, (long) length2);
      for (long j = jStart; j<=jStop; ++j) {
        if (fabs(series1_i-series2[j-1])<epsilon) { matrix[i][j] = matrix[i-1][j-1]+1; }
        else {
          // Because of the window, we MUST include the diagonal! Imagine the case with w=0,
          // the cost could not be propagated without looking at the diagonal.
          matrix[i][j] = std::max(matrix[i-1][j-1], std::max(matrix[i][j-1], matrix[i-1][j]));
        }
      }
    }

    // Check if we have an alignment
    if (matrix[length1][length2]==-1) { return PINF; }

    // Convert in range [0-1]
    return 1.0-(((double) matrix[length1][length2])/(double) minLength);
  }

}

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
TEST_CASE("Univariate LCSS Fixed length", "[lcss][univariate]") {
  // Setup univariate with fixed length
  Mocker mocker;
  const auto& wratios = mocker.wratios;
  const auto& epsilons = mocker.epsilons;
  const auto fset = mocker.vec_randvec(nbitems);

  SECTION("LCSS(s,s) == 0") {
    for (const auto& s: fset) {
      for (double e: epsilons) {
        for (double wr: wratios) {
          auto w = (size_t) (wr*mocker._fixl);
          INFO("epsilon = " << e << " w = " << w)
          const double lcss_ref_v = ref::lcss_matrix(s, s, w, e);
          REQUIRE(lcss_ref_v==0);

          const auto lcss_v = lcss(s.size(), s.size(), dist(s, s), w, e, PINF);
          REQUIRE(lcss_v==0);
        }
      }
    }
  }

  SECTION("LCSS(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i+1];
      for (double e: epsilons) {
        for (double wr: wratios) {
          const auto w = (size_t) (wr*mocker._fixl);
          const double lcss_ref_v = ref::lcss_matrix(s1, s2, w, e);
          INFO("Exact same operation order. Expect exact floating point equality.")
          const auto lcss_tempo = lcss(s1.size(), s2.size(), dist(s1, s2), w, e, PINF);
          REQUIRE(lcss_ref_v==lcss_tempo);
        }
      }
    }
  }

  SECTION("NN1 LCSS") {
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

      // NN1 loop
      for (size_t j = 0; j<nbitems; j += 5) {
        // Skip self.
        if (i==j) { continue; }
        const auto& s2 = fset[j];
        for (double e: epsilons) {
          for (double wr: wratios) {
            const auto w = (size_t) (wr*mocker._fixl);

            // --- --- --- --- --- --- --- --- --- --- --- ---
            const double v_ref = ref::lcss_matrix(s1, s2, w, e);
            if (v_ref<bsf_ref) {
              idx_ref = j;
              bsf_ref = v_ref;
            }

            // --- --- --- --- --- --- --- --- --- --- --- ---
            const auto v = lcss(s1.size(), s2.size(), dist(s1, s2), w, e, PINF);
            if (v<bsf) {
              idx = j;
              bsf = v;
            }

            REQUIRE(idx_ref==idx);

            // --- --- --- --- --- --- --- --- --- --- --- ---
            const auto v_tempo = lcss(s1.size(), s2.size(), dist(s1, s2), w, e, bsf_tempo);
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

TEST_CASE("Univariate LCSS Variable length", "[lcss][univariate]") {
  // Setup univariate dataset with varying length
  Mocker mocker;
  const auto& wratios = mocker.wratios;
  const auto& epsilons = mocker.epsilons;
  const auto fset = mocker.vec_rs_randvec(nbitems);

  SECTION("LCSS(s,s) == 0") {
    for (const auto& s: fset) {
      for (double e: epsilons) {
        for (double wr: wratios) {
          const auto w = (size_t) (wr*(s.size()));
          const double lcss_ref_v = ref::lcss_matrix(s, s, w, e);
          REQUIRE(lcss_ref_v==0);
          const auto lcss_v = lcss(s.size(), s.size(), dist(s, s), w, e, PINF);
          REQUIRE(lcss_v==0);
        }
      }
    }
  }

  SECTION("LCSS(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      for (size_t ie = 0; ie<epsilons.size(); ++ie) {
        double e = epsilons[ie];
        for (size_t iw = 2; iw<wratios.size(); ++iw) {
          double wr = wratios[iw];
          const auto& s1 = fset[i];
          const auto& s2 = fset[i+1];
          const auto w = (size_t) (wr*(min(s1.size(), s2.size())));
          const double lcss_ref_v = ref::lcss_matrix(s1, s2, w, e);
          INFO("Exact same operation order. Expect exact floating point equality.")
          INFO(i << " " << ie << " " << iw)
          const auto lcss_tempo_v = lcss(s1.size(), s2.size(), dist(s1, s2), w, e, PINF);
          REQUIRE(lcss_ref_v==lcss_tempo_v);
        }
      }
    }
  }

  SECTION("NN1 LCSS") {
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

      // NN1 loop
      for (size_t j = 0; j<nbitems; j += 5) {
        // Skip self.
        if (i==j) { continue; }
        const auto& s2 = fset[j];
        for (double e: epsilons) {
          for (double wr: wratios) {
            const auto w = (size_t) (wr*(min(s1.size(), s2.size())));
            // --- --- --- --- --- --- --- --- --- --- --- ---
            const double v_ref = ref::lcss_matrix(s1, s2, w, e);
            if (v_ref<bsf_ref) {
              idx_ref = j;
              bsf_ref = v_ref;
            }
            // --- --- --- --- --- --- --- --- --- --- --- ---
            const auto v = lcss(s1.size(), s2.size(), dist(s1, s2), w, e, PINF);
            if (v<bsf) {
              idx = j;
              bsf = v;
            }
            REQUIRE(idx_ref==idx);
            // --- --- --- --- --- --- --- --- --- --- --- ---
            const auto v_tempo = lcss(s1.size(), s2.size(), dist(s1, s2), w, e, bsf_tempo);
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