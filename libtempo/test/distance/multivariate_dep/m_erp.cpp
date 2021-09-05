#define CATCH_CONFIG_FAST_COMPILE

#include <catch.hpp>
#include <libtempo/distance/erp.hpp>
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

  double erp_matrix(const vector<double>& a, const vector<double>& b, size_t dim, const vector<double>& gv, size_t w_) {
    // Length of the series depends on the actual size of the data and the dimension
    const long la = (long) a.size()/(long) dim;
    const long lb = (long) b.size()/(long) dim;
    long w = (long) w_;

    // Check lengths. Be explicit in the conditions.
    if (la==0 && lb==0) { return 0; }
    if (la==0 && lb!=0) { return PINF<double>; }
    if (la!=0 && lb==0) { return PINF<double>; }

    // Cap the windows
    if (w>la) { w = la; }

    // Check if, given the constralong w, we can have an alignment.
    if (la-lb>w) { return PINF<double>; }

    // Allocate a double buffer for the columns. Declare the index of the 'c'urrent and 'p'revious buffer.
    // Note: we use a vector as a way to initialize the buffer with PINF<double>
    vector<std::vector<double>> matrix(la+1, std::vector<double>(lb+1, PINF<double>));

    // Initialisation of the first line and column
    matrix[0][0] = 0;
    for (long j{1}; j<lb+1; j++) {
      matrix[0][j] = matrix[0][j-1]+sqedN(gv, 0, b, j-1, dim);
    }
    for (long i{1}; i<la+1; i++) {
      matrix[i][0] = matrix[i-1][0]+sqedN(a, i-1, gv, 0, dim);
    }

    // Iterate over the lines
    for (long i{1}; i<la+1; ++i) {
      long l = max<long>(i-w, 1);
      long r = min<long>(i+w+1, lb+1);

      // Iterate through the rest of the columns
      for (long j{l}; j<r; ++j) {
        matrix[i][j] = min(
          matrix[i][j-1]+sqedN(gv, 0, b, j-1, dim),         // Previous
          min(matrix[i-1][j-1]+sqedN(a, i-1, b, j-1, dim),  // Diagonal
            matrix[i-1][j]+sqedN(a, i-1, gv, 0, dim)        // Above
          )
        );
      }
    } // End of for over lines

    return matrix[la][lb];
  }

}

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
TEST_CASE("Multivariate Dependent ERP Fixed length", "[erp][multivariate]") {
  // Setup univariate with fixed length
  Mocker mocker;
  mocker._dim = ndim;
  const auto& wratios = mocker.wratios;
  const auto& gvalues = mocker.gvalues;

  const auto fset = mocker.vec_randvec(nbitems);

  SECTION("ERP(s,s) == 0") {
    for (const auto& s: fset) {
      for (double wr: wratios) {
        auto w = (size_t) (wr*mocker._fixl);

        for (auto gv_: gvalues) {
          vector<double> gv{ndim, gv_};

          const double dtw_ref_v = erp_matrix(s, s, ndim, gv, w);
          REQUIRE(dtw_ref_v==0);

          const auto dtw_v = erp<double>(s, s, ndim, gv, w);
          REQUIRE(dtw_v==0);
        }
      }
    }
  }

  SECTION("ERP(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i+1];

      for (double wr: wratios) {
        const auto w = (size_t) (wr*mocker._fixl);

        for (auto gv_: gvalues) {
          vector<double> gv{ndim, gv_};

          const double dtw_ref_v = erp_matrix(s1, s2, ndim, gv, w);
          INFO("Exact same operation order. Expect exact floating point equality.")

          const auto dtw_eap_v = erp<double>(s1, s2, ndim, gv, w);
          REQUIRE(dtw_ref_v==dtw_eap_v);
        }
      }
    }
  }

  SECTION("NN1 ERP") {
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

        for (double wr: wratios) {
          const auto w = (size_t) (wr*mocker._fixl);
          for (auto gv_: gvalues) {
            vector<double> gv{ndim, gv_};

            // --- --- --- --- --- --- --- --- --- --- --- ---
            const double v_ref = erp_matrix(s1, s2, ndim, gv, w);
            if (v_ref<bsf_ref) {
              idx_ref = j;
              bsf_ref = v_ref;
            }

            // --- --- --- --- --- --- --- --- --- --- --- ---
            const auto v = erp<double>(s1, s2, ndim, gv, w);
            if (v<bsf) {
              idx = j;
              bsf = v;
            }

            REQUIRE(idx_ref==idx);

            // --- --- --- --- --- --- --- --- --- --- --- ---
            const auto v_eap = erp<double>(s1, s2, ndim, gv, w);
            if (v_eap<bsf_eap) {
              idx_eap = j;
              bsf_eap = v_eap;
            }

            REQUIRE(idx_ref==idx_eap);
          }
        }
      }
    }// End query loop
  }// End section
}

TEST_CASE("Multivariate Dependent ERP Variable length", "[erp][multivariate]") {
  // Setup univariate dataset with varying length
  Mocker mocker;
  const auto& wratios = mocker.wratios;
  const auto& gvalues = mocker.gvalues;
  const auto fset = mocker.vec_rs_randvec(nbitems);

  SECTION("ERP(s,s) == 0") {
    for (const auto& s: fset) {
      for (double wr: wratios) {
        const auto w = (size_t) (wr*(s.size()));

        for (auto gv_: gvalues) {
          vector<double> gv{ndim, gv_};

          const double dtw_ref_v = erp_matrix(s, s, ndim, gv, w);
          REQUIRE(dtw_ref_v==0);

          const auto dtw_v = erp<double>(s, s, ndim, gv, w);
          REQUIRE(dtw_v==0);
        }
      }
    }
  }

  SECTION("ERP(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      for (double wr: wratios) {
        const auto& s1 = fset[i];
        const auto& s2 = fset[i+1];
        const auto w = (size_t) (wr*(min(s1.size(), s2.size())));

        for (auto gv_: gvalues) {
          vector<double> gv{ndim, gv_};

          const double dtw_ref_v = erp_matrix(s1, s2, ndim, gv, w);
          INFO("Exact same operation order. Expect exact floating point equality.")

          const auto dtw_eap_v = erp<double>(s1, s2, ndim, gv, w);
          REQUIRE(dtw_ref_v==dtw_eap_v);
        }
      }
    }
  }

  SECTION("NN1 ERP") {
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

        for (double wr: wratios) {
          const auto w = (size_t) (wr*(min(s1.size(), s2.size())));

          for (auto gv_: gvalues) {
            vector<double> gv{ndim, gv_};

            // --- --- --- --- --- --- --- --- --- --- --- ---
            const double v_ref = erp_matrix(s1, s2, ndim, gv, w);
            if (v_ref<bsf_ref) {
              idx_ref = j;
              bsf_ref = v_ref;
            }

            // --- --- --- --- --- --- --- --- --- --- --- ---
            const auto v = erp<double>(s1, s2, ndim, gv, w);
            if (v<bsf) {
              idx = j;
              bsf = v;
            }

            REQUIRE(idx_ref==idx);

            // --- --- --- --- --- --- --- --- --- --- --- ---
            const auto v_eap = erp<double>(s1, s2, ndim, gv, w, bsf_eap);
            if (v_eap<bsf_eap) {
              idx_eap = j;
              bsf_eap = v_eap;
            }

            REQUIRE(idx_ref==idx_eap);

          }
        }
      }
    }// End query loop
  }// End section

}
