#define CATCH_CONFIG_FAST_COMPILE

#include <catch.hpp>
#include <libtempo/distance/cdtw.hpp>
#include <iostream>

#include "../mock/mockseries.hpp"

using namespace mock;
using namespace libtempo::distance;
constexpr size_t nbitems = 500;
constexpr size_t ndim = 3;

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

  double cdtw_matrix(const vector<double>& a, const vector<double>& b, size_t dim, size_t w_) {
    // Length of the series depends on the actual size of the data and the dimension
    const long la = (long) a.size()/(long) dim;
    const long lb = (long) b.size()/(long) dim;
    const long w = (long) w_;

    // Check lengths. Be explicit in the conditions.
    if (la==0 && lb==0) { return 0; }
    if (la==0 && lb!=0) { return PINF<double>; }
    if (la!=0 && lb==0) { return PINF<double>; }

    // Allocate the working space: full matrix + space for borders (first column / first line)
    size_t msize = max(la, lb)+1;
    vector<std::vector<double>> matrix(msize, std::vector<double>(msize, PINF<double>));

    // Initialisation (all the matrix is initialised at +INF)
    matrix[0][0] = 0;

    // For each line
    // Note: series1 and series2 are 0-indexed while the matrix is 1-indexed (0 being the borders)
    //       hence, we have i-1 and j-1 when accessing series1 and series2
    for (long i = 1; i<=(long) la; i++) {
      long jStart = max<long>(1, i-w);
      long jStop = min<long>(i+w, lb);
      for (long j = jStart; j<=jStop; j++) {
        double prev = matrix[i][j-1];
        double diag = matrix[i-1][j-1];
        double top = matrix[i-1][j];
        matrix[i][j] = min(prev, std::min(diag, top))+sqedN(a, i-1, b, j-1, dim);
      }
    }

    return matrix[la][lb];
  }

}

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
TEST_CASE("Multivariate Dependent CDTW Fixed length", "[cdtw][multivariate]") {
  // Setup univariate with fixed length
  Mocker mocker;
  mocker._dim = ndim;
  const auto& wratios = mocker.wratios;

  const auto fset = mocker.vec_randvec(nbitems);

  SECTION("DTW(s,s) == 0") {
    for (const auto& s: fset) {
      for (double wr:wratios) {
        auto w = (size_t) (wr*mocker._fixl);

        const double dtw_ref_v = cdtw_matrix(s, s, ndim, w);
        REQUIRE(dtw_ref_v==0);

        const auto dtw_v = cdtw<double>(s, s, ndim, w);
        REQUIRE(dtw_v==0);
      }
    }
  }

  SECTION("DTW(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i+1];

      for (double wr:wratios) {
        const auto w = (size_t) (wr*mocker._fixl);

        const double dtw_ref_v = cdtw_matrix(s1, s2, ndim, w);
        INFO("Exact same operation order. Expect exact floating point equality.")

        const auto dtw_eap_v = cdtw<double>(s1, s2, ndim, w);
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

        for (double wr:wratios) {
          const auto w = (size_t) (wr*mocker._fixl);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const double v_ref = cdtw_matrix(s1, s2, ndim, w);
          if (v_ref<bsf_ref) {
            idx_ref = j;
            bsf_ref = v_ref;
          }

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v = cdtw<double>(s1, s2, ndim, w);
          if (v<bsf) {
            idx = j;
            bsf = v;
          }

          REQUIRE(idx_ref==idx);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v_eap = cdtw<double>(s1, s2, ndim, w, bsf_eap);
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

TEST_CASE("Multivariate Dependent CDTW Variable length", "[cdtw][multivariate]") {
  // Setup univariate dataset with varying length
  Mocker mocker;
  const auto& wratios = mocker.wratios;
  const auto fset = mocker.vec_rs_randvec(nbitems);

  SECTION("DTW(s,s) == 0") {
    for (const auto& s: fset) {
      for (double wr:wratios) {
        const auto w = (size_t) (wr*(s.size()));
        const double dtw_ref_v = cdtw_matrix(s, s, ndim, w);
        REQUIRE(dtw_ref_v==0);

        const auto dtw_v = cdtw<double>(s, s, ndim, w);
        REQUIRE(dtw_v==0);
      }
    }
  }

  SECTION("DTW(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      for (double wr:wratios) {
        const auto& s1 = fset[i];
        const auto& s2 = fset[i+1];
        const auto w = (size_t) (wr*(min(s1.size(), s2.size())));

        const double dtw_ref_v = cdtw_matrix(s1, s2, ndim, w);
        INFO("Exact same operation order. Expect exact floating point equality.")

        const auto dtw_eap_v = cdtw<double>(s1, s2, ndim, w);
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

        for (double wr:wratios) {
          const auto w = (size_t) (wr*(min(s1.size(), s2.size())));

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const double v_ref = cdtw_matrix(s1, s2, ndim, w);
          if (v_ref<bsf_ref) {
            idx_ref = j;
            bsf_ref = v_ref;
          }

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v = cdtw<double>(s1, s2, ndim, w);
          if (v<bsf) {
            idx = j;
            bsf = v;
          }

          REQUIRE(idx_ref==idx);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v_eap = cdtw<double>(s1, s2, ndim, w, bsf_eap);
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
