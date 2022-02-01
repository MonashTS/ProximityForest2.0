#define CATCH_CONFIG_FAST_COMPILE

#include <catch.hpp>
#include <libtempo/distance/lcss.hpp>
#include <iostream>

#include <mock/mockseries.hpp>

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

  /// Naive Univariate LCSS with a window. Reference code. Use squared euclidean distance instead of L1.
  double lcss_matrix_uni(const vector<double>& series1, const vector<double>& series2, double epsilon, size_t w_) {
    constexpr double PINF = libtempo::utils::PINF<double>;
    size_t length1 = series1.size();
    size_t length2 = series2.size();
    long w = (long) w_;

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
    //       hence, we have i-1 and j-1 when accessing series1 and series2
    for (long i = 1; i<=(long) length1; ++i) {
      auto series1_i = series1[i-1];
      long jStart = max<long>(1, i-w);
      long jStop = min<long>(i+w, (long) length2);
      for (long j = jStart; j<=jStop; ++j) {
        // Warning: comparison with LCSS using multivariate with dim=1: use squared L2 norm
        if (sqdist(series1_i, series2[j-1])<epsilon) { matrix[i][j] = matrix[i-1][j-1]+1; }
        else {
          // Because of the window, we MUST include the diagonal! Imagine the case with w=0,
          // the cost could not be propagated without looking at the diagonal.
          matrix[i][j] = std::max(matrix[i-1][j-1], std::max(matrix[i][j-1], matrix[i-1][j]));
        }
      }
    }

    // Check if we have an alignment
    if (matrix[length1][length2]==-1) { return libtempo::utils::PINF<double>; }

    // Convert in range [0-1]
    return 1.0-(((double) matrix[length1][length2])/(double) minLength);
  }

  /// Naive Multivariate LCSS with a window. Reference code.
  double lcss_matrix(const vector<double>& sa, const vector<double>& sb, size_t dim, double epsilon, size_t w_) {
    // Length of the series depends on the actual size of the data and the dimension
    const long la = (long) sa.size()/(long) dim;
    const long lb = (long) sb.size()/(long) dim;
    const long w = (long) w_;

    // Allocate the working space: full matrix + space for borders (first column / first line)
    size_t maxLength = max<size_t>(la, lb);
    size_t minLength = min<size_t>(la, lb);
    vector<std::vector<int>> matrix(maxLength+1, std::vector<int>(maxLength+1, 0));

    // Marker for final point
    matrix[la][lb] = -1;

    // LCSS
    // Note: sa and sb are 0-indexed while the matrix is 1-indexed (0 being the borders)
    //       hence, we have i-1 and j-1 when accessing sa and sb
    for (long i = 1; i<=(long) la; ++i) {
      long jStart = max<long>(1, i-w);
      long jStop = min<long>(i+w, (long) lb);
      for (long j = jStart; j<=jStop; ++j) {
        // Warning: comparison with LCSS using multivariate with dim=1: use squared L2 norm
        if (sqedN(sa, i-1, sb, j-1, dim)<epsilon) { matrix[i][j] = matrix[i-1][j-1]+1; }
        else {
          // Because of the window, we MUST include the diagonal! Imagine the case with w=0,
          // the cost could not be propagated without looking at the diagonal.
          matrix[i][j] = std::max(matrix[i-1][j-1], std::max(matrix[i][j-1], matrix[i-1][j]));
        }
      }
    }

    // Check if we have an alignment
    if (matrix[la][lb]==-1) { return libtempo::utils::PINF<double>; }

    // Convert in range [0-1]
    return 1.0-(((double) matrix[la][lb])/(double) minLength);
  }
}

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
TEST_CASE("Multivariate Dependent LCSS Fixed length", "[lcss][multivariate]") {
  // Setup univariate with fixed length
  Mocker mocker;
  mocker._dim = ndim;
  const auto& wratios = mocker.wratios;
  const auto& epsilons = mocker.epsilons;
  const auto fset = mocker.vec_randvec(nbitems);

  SECTION("LCSS(s,s) == 0") {
    for (const auto& s: fset) {
      for (double e: epsilons) {
        for (double wr: wratios) {
          auto w = (size_t) (wr*mocker._fixl);

          const double lcss_ref_v = lcss_matrix(s, s, ndim, e, w);
          REQUIRE(lcss_ref_v==0);

          const auto lcss_v = lcss<double>(s, s, ndim, e, w);
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

          // Check Uni
          {
            const double lcss_ref_v = lcss_matrix(s1, s2, 1, e, w);
            const double lcss_ref_uni_v = lcss_matrix_uni(s1, s2, e, w);
            const auto lcss_tempo_v = lcss<double>(s1, s2, 1, e, w);
            REQUIRE(lcss_ref_v==lcss_ref_uni_v);
            REQUIRE(lcss_ref_v==lcss_tempo_v);
          }

          // Check Multi
          {
            const double lcss_ref_v = lcss_matrix(s1, s2, ndim, e, w);
            INFO("Exact same operation order. Expect exact floating point equality.")
            const auto lcss_tempo_v = lcss<double>(s1, s2, ndim, e, w);
            REQUIRE(lcss_ref_v==lcss_tempo_v);
          }
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
        for (double e: epsilons) {
          for (double wr: wratios) {
            const auto w = (size_t) (wr*mocker._fixl);
            // --- --- --- --- --- --- --- --- --- --- --- ---
            const double v_ref = lcss_matrix(s1, s2, ndim, e, w);
            if (v_ref<bsf_ref) {
              idx_ref = j;
              bsf_ref = v_ref;
            }
            // --- --- --- --- --- --- --- --- --- --- --- ---
            const auto v = lcss<double>(s1, s2, ndim, e, w);
            if (v<bsf) {
              idx = j;
              bsf = v;
            }
            REQUIRE(idx_ref==idx);
            // --- --- --- --- --- --- --- --- --- --- --- ---
            const auto v_tempo = lcss<double>(s1, s2, ndim, e, w, bsf_tempo);
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

TEST_CASE("Multivariate Dependent LCSS Variable length", "[lcss][multivariate]") {
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
          const double lcss_ref_v = lcss_matrix(s, s, ndim, e, w);
          REQUIRE(lcss_ref_v==0);
          const auto lcss_v = lcss<double>(s, s, ndim, e, w);
          REQUIRE(lcss_v==0);
        }
      }
    }
  }

  SECTION("LCSS(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      for (double e: epsilons) {
        for (double wr: wratios) {
          const auto& s1 = fset[i];
          const auto& s2 = fset[i+1];
          const auto w = (size_t) (wr*(min(s1.size(), s2.size())));

          // Check Uni
          {
            const double lcss_ref_v = lcss_matrix(s1, s2, 1, e, w);
            const double lcss_ref_uni_v = lcss_matrix_uni(s1, s2, e, w);
            const auto lcss_tempo_v = lcss<double>(s1, s2, 1, e, w);
            REQUIRE(lcss_ref_v==lcss_ref_uni_v);
            REQUIRE(lcss_ref_v==lcss_tempo_v);
          }

          // Check Multi
          {
            const double lcss_ref_v = lcss_matrix(s1, s2, ndim, e, w);
            INFO("Exact same operation order. Expect exact floating point equality.")

            const auto lcss_tempo_v = lcss<double>(s1, s2, ndim, e, w);
            REQUIRE(lcss_ref_v==lcss_tempo_v);
          }
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
        for (double e: epsilons) {
          for (double wr: wratios) {
            const auto w = (size_t) (wr*(min(s1.size(), s2.size())));
            // --- --- --- --- --- --- --- --- --- --- --- ---
            const double v_ref = lcss_matrix(s1, s2, ndim, e, w);
            if (v_ref<bsf_ref) {
              idx_ref = j;
              bsf_ref = v_ref;
            }
            // --- --- --- --- --- --- --- --- --- --- --- ---
            const auto v = lcss<double>(s1, s2, ndim, e, w);
            if (v<bsf) {
              idx = j;
              bsf = v;
            }
            REQUIRE(idx_ref==idx);
            // --- --- --- --- --- --- --- --- --- --- --- ---
            const auto v_tempo = lcss<double>(s1, s2, ndim, e, w, bsf_tempo);
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
