#define CATCH_CONFIG_FAST_COMPILE

#include <catch.hpp>
#include <mock/mockseries.hpp>

#include <libtempo/distance/dtw.hpp>

using namespace mock;
using namespace libtempo::distance;
using namespace libtempo::distance::multivariate;
constexpr size_t nbitems = 500;
constexpr size_t ndim = 3;
constexpr double INF = libtempo::utils::PINF<double>;

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Reference
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
namespace {

  using namespace libtempo::utils;
  using namespace std;

  /// Naive Univariate DTW without a window. Reference code.
  double dtw_matrix_uni(const vector<double>& series1, const vector<double>& series2) {
    const auto length1 = series1.size();
    const auto length2 = series2.size();
    // Check lengths. Be explicit in the conditions.
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

  /// Naive Multivariate DTW without a window. Reference code.
  double dtw_matrix(const vector<double>& a, const vector<double>& b, size_t dim) {
    // Length of the series depends on the actual size of the data and the dimension
    const auto la = a.size()/dim;
    const auto lb = b.size()/dim;

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
    for (size_t i = 1; i<=la; i++) {
      for (size_t j = 1; j<=lb; j++) {
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
TEST_CASE("Multivariate Dependent DTW Fixed length", "[dtw][multivariate]") {
  // Setup univariate with fixed length
  Mocker mocker;
  mocker._dim = ndim;

  const auto l = mocker._fixl;
  const auto l1 = l * mocker._dim;

  const auto fset = mocker.vec_randvec(nbitems);

  SECTION("DTW(s,s) == 0") {
    for (const auto &s: fset) {
      const double dtw_ref_v = dtw_matrix(s, s, ndim);
      REQUIRE(dtw_ref_v == 0);

      const auto dtw_v = dtw<double>(l, l, ad2N<double>(s, s, ndim), INF);
      REQUIRE(dtw_v == 0);
    }
  }

  SECTION("DTW(s1, s2)") {
     for (size_t i = 0; i<nbitems-1; ++i) {
       const auto& s1 = fset[i];
       const auto& s2 = fset[i+1];

       // Check Uni
       {
         const double dtw_ref_v = dtw_matrix(s1, s2, 1);
         const double dtw_ref_uni_v = dtw_matrix_uni(s1, s2);
         const auto dtw_tempo_v = dtw<double>(l1, l1, ad2N<double>(s1, s2, 1), INF);
         REQUIRE(dtw_ref_v==dtw_ref_uni_v);
         REQUIRE(dtw_ref_v==dtw_tempo_v);
       }

       // Check Multi
       {
         const double dtw_ref_v = dtw_matrix(s1, s2, ndim);
         INFO("Exact same operation order. Expect exact floating point equality.")

         const auto dtw_tempo_v = dtw<double>(l, l, ad2N<double>(s1, s2, ndim), INF);
         REQUIRE(dtw_ref_v==dtw_tempo_v);
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
       size_t idx_tempo = 0;
       double bsf_tempo = lu::PINF<double>;

       // NN1 loop
       for (size_t j = 0; j<nbitems; j += 5) {
         // Skip self.
         if (i==j) { continue; }
         const auto& s2 = fset[j];

         // --- --- --- --- --- --- --- --- --- --- --- ---
         const double v_ref = dtw_matrix(s1, s2, ndim);
         if (v_ref<bsf_ref) {
           idx_ref = j;
           bsf_ref = v_ref;
         }

         // --- --- --- --- --- --- --- --- --- --- --- ---
         const auto v = dtw<double>(l, l, ad2N<double>(s1, s2, ndim), INF);
         if (v<bsf) {
           idx = j;
           bsf = v;
         }

         REQUIRE(idx_ref==idx);

         // --- --- --- --- --- --- --- --- --- --- --- ---
         const auto v_tempo = dtw<double>(l, l, ad2N<double>(s1, s2, ndim), bsf_tempo);
         if (v_tempo<bsf_tempo) {
           idx_tempo = j;
           bsf_tempo = v_tempo;
         }

         REQUIRE(idx_ref==idx_tempo);
       }
     }// End query loop
   }// End section
 }


TEST_CASE("Multivariate Dependent DTW Variable length", "[dtw][multivariate]") {
  // Setup univariate dataset with varying length
  Mocker mocker;
  const auto ndim = mocker._dim;
  const auto fset = mocker.vec_rs_randvec(nbitems);

  auto ld = [ndim](const std::vector<double>& v){
    return v.size()/ndim;
  };


  SECTION("DTW(s,s) == 0") {
    for (const auto& s: fset) {
      const double dtw_ref_v = dtw_matrix(s, s, ndim);
      REQUIRE(dtw_ref_v==0);

      const auto dtw_v = dtw<double>(ld(s), ld(s), ad2N<double>(s, s, ndim), INF);
      REQUIRE(dtw_v==0);
    }
  }

  SECTION("DTW(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i+1];

      // Check Uni
      {
        const double dtw_ref_v = dtw_matrix(s1, s2, 1);
        const double dtw_ref_uni_v = dtw_matrix_uni(s1, s2);
        const auto dtw_tempo_v = dtw<double>(s1.size(), s2.size(), ad2N<double>(s1, s2, 1), INF);
        REQUIRE(dtw_ref_v==dtw_ref_uni_v);
        REQUIRE(dtw_ref_v==dtw_tempo_v);
      }

      // Check Multi
      {
        const double dtw_ref_v = dtw_matrix(s1, s2, ndim);
        INFO("Exact same operation order. Expect exact floating point equality.")

        const auto dtw_tempo_v = dtw<double>(ld(s1), ld(s2), ad2N<double>(s1, s2, ndim), INF);
        REQUIRE(dtw_ref_v==dtw_tempo_v);
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
      size_t idx_tempo = 0;
      double bsf_tempo = lu::PINF<double>;

      // NN1 loop
      for (size_t j = 0; j<nbitems; j += 5) {
        // Skip self.
        if (i==j) { continue; }
        const auto& s2 = fset[j];

        // --- --- --- --- --- --- --- --- --- --- --- ---
        const double v_ref = dtw_matrix(s1, s2, ndim);
        if (v_ref<bsf_ref) {
          idx_ref = j;
          bsf_ref = v_ref;
        }

        // --- --- --- --- --- --- --- --- --- --- --- ---
        const auto v = dtw<double>(ld(s1), ld(s2), ad2N<double>(s1, s2, ndim), INF);
        if (v<bsf) {
          idx = j;
          bsf = v;
        }

        REQUIRE(idx_ref==idx);

        // --- --- --- --- --- --- --- --- --- --- --- ---
        const auto v_tempo = dtw<double>(ld(s1), ld(s2), ad2N<double>(s1, s2, ndim), bsf_tempo);
        if (v_tempo<bsf_tempo) {
          idx_tempo = j;
          bsf_tempo = v_tempo;
        }

        REQUIRE(idx_ref==idx_tempo);
      }
    }// End query loop
  }// End section

}