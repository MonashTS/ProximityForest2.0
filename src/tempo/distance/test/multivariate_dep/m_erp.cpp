#define CATCH_CONFIG_FAST_COMPILE

#include <mock/mockseries.hpp>
#include <catch.hpp>
#include <tempo/distance/elastic/erp.hpp>

using namespace mock;
using namespace tempo::distance;
using namespace tempo::distance::multivariate;
constexpr size_t nbitems = 500;
constexpr size_t ndim = 3;
constexpr double INF = tempo::utils::PINF<double>;

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Reference
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
namespace {

  using namespace tempo;
  using namespace tempo::utils;
  using namespace std;

  /// Naive Multivariate ERP with a window. Reference code.
  double erp_matrix_uni(const vector<double>& series1, const vector<double>& series2, double gValue, long w) {
    const long length1 = to_signed(series1.size());
    const long length2 = to_signed(series2.size());

    // Check lengths. Be explicit in the conditions.
    if (length1==0 && length2==0) { return 0; }
    if (length1==0 && length2!=0) { return PINF<double>; }
    if (length1!=0 && length2==0) { return PINF<double>; }

    // We will only allocate a double-row buffer: use the smallest possible dimension as the columns.
    const vector<double>& cols = (length1<length2) ? series1 : series2;
    const vector<double>& lines = (length1<length2) ? series2 : series1;
    const long nbcols = min(length1, length2);
    const long nblines = max(length1, length2);

    // Cap the windows
    if (w>nblines) { w = nblines; }

    // Check if, given the constralong w, we can have an alignment.
    if (nblines-nbcols>w) { return PINF<double>; }

    // Allocate a double buffer for the columns. Declare the index of the 'c'urrent and 'p'revious buffer.
    // Note: we use a vector as a way to initialize the buffer with PINF<double>
    vector<std::vector<double>> matrix(nblines+1, std::vector<double>(nbcols+1, PINF<double>));

    // Initialisation of the first line and column
    matrix[0][0] = 0;
    for (long j{1}; j<nbcols+1; j++) {
      matrix[0][j] = matrix[0][j-1]+sqdist(gValue, cols[j-1]);
    }
    for (long i{1}; i<nblines+1; i++) {
      matrix[i][0] = matrix[i-1][0]+sqdist(lines[i-1], gValue);
    }

    // Iterate over the lines
    for (long i{1}; i<nblines+1; ++i) {
      const double li = lines[i-1];
      long l = max<long>(i-w, 1);
      long r = min<long>(i+w+1, nbcols+1);

      // Iterate through the rest of the columns
      for (long j{l}; j<r; ++j) {
        matrix[i][j] = min(
          matrix[i][j-1]+sqdist(gValue, cols[j-1]),        // Previous
          min(matrix[i-1][j-1]+sqdist(li, cols[j-1]),    // Diagonal
            matrix[i-1][j]+sqdist(li, gValue)              // Above
          )
        );
      }
    } // End of for over lines

    return matrix[nblines][nbcols];
  }

  /// Naive Univariate ERP with a window. Reference code.
  double erp_matrix(const vector<double>& a, const vector<double>& b, size_t dim, const vector<double>& gv, size_t w_) {
    // Length of the series depends on the actual size of the data and the dimension
    const long la = (long) a.size()/(long) dim;
    const long lb = (long) b.size()/(long) dim;
    long w = (long) w_;

    // Check lengths. Be explicit in the conditions.
    if (la==0 && lb==0) { return 0; }
    if (la==0 && lb!=0) { return PINF<double>; }
    if (la!=0 && lb==0) { return PINF<double>; }

    // We will only allocate a double-row buffer: use the smallest possible dimension as the columns.
    const vector<double>& cols = (la<lb) ? a : b;
    const vector<double>& lines = (la<lb) ? b : a;
    const long nbcols = min(la, lb);
    const long nblines = max(la, lb);

    // Cap the windows
    if (w>nblines) { w = nblines; }

    // Check if, given the constralong w, we can have an alignment.
    if (nblines-nbcols>w) { return PINF<double>; }

    // Allocate a double buffer for the columns. Declare the index of the 'c'urrent and 'p'revious buffer.
    // Note: we use a vector as a way to initialize the buffer with PINF<double>
    vector<std::vector<double>> matrix(nblines+1, std::vector<double>(nbcols+1, PINF<double>));

    // Initialisation of the first line and column
    matrix[0][0] = 0;
    for (long j{1}; j<nbcols+1; j++) {
      matrix[0][j] = matrix[0][j-1]+sqedN(gv, 0, cols, j-1, dim);
    }
    for (long i{1}; i<nblines+1; i++) {
      matrix[i][0] = matrix[i-1][0]+sqedN(lines, i-1, gv, 0, dim);
    }

    // Iterate over the lines
    for (long i{1}; i<nblines+1; ++i) {
      long l = max<long>(i-w, 1);
      long r = min<long>(i+w+1, nbcols+1);

      // Iterate through the rest of the columns
      for (long j{l}; j<r; ++j) {
        matrix[i][j] = min(
          matrix[i][j-1]+sqedN(gv, 0, cols, j-1, dim),              // Previous
          min(matrix[i-1][j-1]+sqedN(lines, i-1, cols, j-1, dim),   // Diagonal
            matrix[i-1][j]+sqedN(lines, i-1, gv, 0, dim)            // Above
          )
        );
      }
    } // End of for over lines

    return matrix[nblines][nbcols];
  }

}

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
TEST_CASE("Multivariate Dependent ERP Fixed length", "[erp][multivariate]") {
  // Setup univariate with fixed length
  Mocker mocker;
  mocker._dim = ndim;
  const auto l = mocker._fixl;
  const auto l1 = l * mocker._dim;
  const auto &wratios = mocker.wratios;
  const auto &gvalues = mocker.gvalues;

  const auto fset = mocker.vec_randvec(nbitems);


  SECTION("ERP(s,s) == 0") {
    for (const auto &s: fset) {
      for (double wr: wratios) {
        auto w = (size_t) (wr * mocker._fixl);

        for (auto gv_: gvalues) {
          vector<double> gv(ndim, gv_);

          const double erp_ref_v = erp_matrix(s, s, ndim, gv, w);
          REQUIRE(erp_ref_v == 0);

          const auto erp_v = erp<double>(l, l, ad2gv(s, gv), ad2gv(s, gv), ad2N<double>(s, s, ndim), w, INF);
          REQUIRE(erp_v == 0);
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
          vector<double> gv(ndim, gv_);

          // Check Uni
          {
            const double erp_ref_v = erp_matrix(s1, s2, 1, gv, w);
            const double erp_ref_uni_v = erp_matrix_uni(s1, s2, gv_, w);
            const auto erp_tempo_uni_v = erp<double>(l1, l1, univariate::ad2gv(s1, gv_), univariate::ad2gv(s2, gv_), ad2N<double>(s1, s2, 1), w, INF);
            REQUIRE(erp_ref_v == erp_ref_uni_v);
            REQUIRE(erp_ref_v == erp_tempo_uni_v);
          }

          // Check Multi
          {
            const double erp_ref_v = erp_matrix(s1, s2, ndim, gv, w);
            INFO("Exact same operation order. Expect exact floating point equality.")

            const auto erp_tempo_v = erp<double>(l, l, ad2gv(s1, gv), ad2gv(s2, gv), ad2N<double>(s1, s2, ndim), w, INF);
            REQUIRE(erp_ref_v==erp_tempo_v);
          }

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
      size_t idx_tempo = 0;
      double bsf_tempo = lu::PINF<double>;

      // NN1 loop
      for (size_t j = 0; j<nbitems; j += 5) {
        // Skip self.
        if (i==j) { continue; }
        const auto& s2 = fset[j];
        // Create the univariate squared Euclidean distance for our erp functions

        for (double wr: wratios) {
          const auto w = (size_t) (wr*mocker._fixl);
          for (auto gv_: gvalues) {
            vector<double> gv(ndim, gv_);

            // --- --- --- --- --- --- --- --- --- --- --- ---
            const double v_ref = erp_matrix(s1, s2, ndim, gv, w);
            if (v_ref<bsf_ref) {
              idx_ref = j;
              bsf_ref = v_ref;
            }

            // --- --- --- --- --- --- --- --- --- --- --- ---
            const auto v = erp<double>(l, l, ad2gv(s1, gv), ad2gv(s2, gv), ad2N<double>(s1, s2, ndim), w, INF);
            if (v<bsf) {
              idx = j;
              bsf = v;
            }

            REQUIRE(idx_ref==idx);

            // --- --- --- --- --- --- --- --- --- --- --- ---
            const auto v_tempo =erp<double>(l, l, ad2gv(s1, gv), ad2gv(s2, gv), ad2N<double>(s1, s2, ndim), w, bsf_tempo);
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

TEST_CASE("Multivariate Dependent ERP Variable length", "[erp][multivariate]") {
  // Setup univariate dataset with varying length
  Mocker mocker;
  const auto &wratios = mocker.wratios;
  const auto &gvalues = mocker.gvalues;
  const auto fset = mocker.vec_rs_randvec(nbitems);

  auto ld = [](const std::vector<double> &v) {
    return v.size() / ndim;
  };

  SECTION("ERP(s,s) == 0") {
    for (const auto &s: fset) {
      for (double wr: wratios) {
        const auto w = (size_t) (wr * (s.size()));

        for (auto gv_: gvalues) {
          vector<double> gv(ndim, gv_);

          const double erp_ref_v = erp_matrix(s, s, ndim, gv, w);
          REQUIRE(erp_ref_v == 0);

          const auto erp_v = erp<double>(ld(s), ld(s), ad2gv(s, gv), ad2gv(s, gv), ad2N<double>(s, s, ndim), w, INF);
          REQUIRE(erp_v == 0);
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
          vector<double> gv(ndim, gv_);

          // Check Uni
          {
            const double erp_ref_v = erp_matrix(s1, s2, 1, gv, w);
            const double erp_ref_uni_v = erp_matrix_uni(s1, s2, gv_, w);
            const auto erp_tempo_uni_v = erp<double>(s1.size(), s2.size(), univariate::ad2gv(s1, gv_), univariate::ad2gv(s2, gv_), ad2N<double>(s1, s2, 1), w, INF);
            REQUIRE(erp_ref_v==erp_ref_uni_v);
            REQUIRE(erp_ref_uni_v==erp_tempo_uni_v);
          }

          // Check Multi
          {
            const double erp_ref_v = erp_matrix(s1, s2, ndim, gv, w);
            INFO("Exact same operation order. Expect exact floating point equality.")

            const auto erp_tempo_v = erp<double>(ld(s1), ld(s2), ad2gv(s1, gv), ad2gv(s2, gv), ad2N<double>(s1, s2, ndim), w, INF);
            REQUIRE(erp_ref_v==erp_tempo_v);
          }

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
      size_t idx_tempo = 0;
      double bsf_tempo = lu::PINF<double>;

      // NN1 loop
      for (size_t j = 0; j<nbitems; j += 5) {
        // Skip self.
        if (i==j) { continue; }
        const auto& s2 = fset[j];

        for (double wr: wratios) {
          const auto w = (size_t) (wr*(min(s1.size(), s2.size())));

          for (auto gv_: gvalues) {
            vector<double> gv(ndim, gv_);

            // --- --- --- --- --- --- --- --- --- --- --- ---
            const double v_ref = erp_matrix(s1, s2, ndim, gv, w);
            if (v_ref<bsf_ref) {
              idx_ref = j;
              bsf_ref = v_ref;
            }

            // --- --- --- --- --- --- --- --- --- --- --- ---
            const auto v = erp<double>(ld(s1), ld(s2), ad2gv(s1, gv), ad2gv(s2, gv), ad2N<double>(s1, s2, ndim), w, INF);
            if (v<bsf) {
              idx = j;
              bsf = v;
            }

            REQUIRE(idx_ref==idx);

            // --- --- --- --- --- --- --- --- --- --- --- ---
            const auto v_tempo = erp<double>(ld(s1), ld(s2), ad2gv(s1, gv), ad2gv(s2, gv), ad2N<double>(s1, s2, ndim), w, bsf_tempo);
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