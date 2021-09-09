#define CATCH_CONFIG_FAST_COMPILE

#include <catch.hpp>
#include <libtempo/distance/msm.hpp>
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

  /// Multivariate Euclidean Distance
  double edN(const vector<double>& a, size_t astart, const vector<double>& b, size_t bstart, size_t dim) {
    double acc{0};
    const size_t aoffset = astart*dim;
    const size_t boffset = bstart*dim;
    for (size_t i{0}; i<dim; ++i) {
      double di = a[aoffset+i]-b[boffset+i];
      acc += di*di;
    }
    return sqrt(acc);
  }

  /// Univariate Squared Euclidean Distance
  double ed(double a, double b) {
    double d = a-b;
    return sqrt(d*d);
  }

  /// Implementing cost from the original paper
  /// c is the minimal cost of an operation
  inline double get_cost(double xnew, double xi, double yj, double cost) {
    // Radius
    double diameter = ed(xi, yj);
    double diameter_ed = std::sqrt(ed(xi, yj));
    double radius = diameter/2;
    double radius_ed = diameter_ed/2;
    double mid = (xi+yj)/2;
    double dist_to_mid = ed(xnew, mid);
    double dist_to_mid_ed = std::sqrt(dist_to_mid);
    bool in_radius = dist_to_mid<=radius;
    //
    if ((xi<=xnew && xnew<=yj) || (xi>=xnew && xnew>=yj)) { return cost; }
    else {
      if (in_radius) {
        std::cerr << "  uni_cost computed but in radius" << std::endl;
        std::cerr << "    sq diameter = " << diameter << std::endl;
        std::cerr << "    sq radius = " << radius << std::endl;
        std::cerr << "    sq dist to mid = " << dist_to_mid << std::endl;
        std::cerr << " --- --- --- --- --- --- --- " << std::endl;
        std::cerr << "    diameter = " << diameter_ed << std::endl;
        std::cerr << "    radius = " << radius_ed << std::endl;
        std::cerr << "    dist to mid = " << dist_to_mid_ed << std::endl;
        std::cerr << " --- --- --- --- --- --- --- " << std::endl;
        std::cerr << "    mid = " << mid << std::endl;
        std::cerr << " --- --- --- --- --- --- --- " << std::endl;
        std::cerr << "    xi = " << xi << std::endl;
        std::cerr << "    xnew = " << xnew << std::endl;
        std::cerr << "    yj = " << yj << std::endl;
        exit(1);
      } else {
        return cost+std::min(ed(xnew, xi), ed(xnew, yj));
      }
    }
  }

  /// MSM cost with multi dimensional hyper sphere
  double msm_cost(
    const vector<double>& X, size_t xnew_start, size_t xi_start,
    const vector<double>& Y, size_t yj_start,
    size_t dim,
    double cost
  ) {
    // Computing the radius of the hyper sphere
    const double radius = edN(X, xi_start, Y, yj_start, dim)/2;
    // Distance to the midpoint
    const size_t xoffset = xi_start*dim;
    const size_t yoffset = yj_start*dim;
    vector<double> mid(dim, 0);
    for (size_t k{0}; k<dim; ++k) { mid[k] = (X[xoffset+k]+Y[yoffset+k])/2; }
    const double dist_to_mid = edN(mid, 0, X, xnew_start, dim);
    // If in the radius, returns the cost
    if (dist_to_mid<=radius) {
      return cost;
    } else {
      const double dist_to_prev = edN(X, xnew_start, X, xi_start, dim);
      const double dist_to_other = edN(X, xnew_start, Y, yj_start, dim);
      return cost+min(dist_to_prev, dist_to_other);
    }
  }

  /// Naive MSM with a window. Reference code.
  double msm_matrix_uni(const vector<double>& series1, const vector<double>& series2, double c) {
    const long length1 = to_signed(series1.size());
    const long length2 = to_signed(series2.size());

    // Check lengths. Be explicit in the conditions.
    if (length1==0 && length2==0) { return 0; }
    if (length1==0 && length2!=0) { return PINF<double>; }
    if (length1!=0 && length2==0) { return PINF<double>; }

    const long maxLength = max(length1, length2);
    vector<std::vector<double>> cost(maxLength, std::vector<double>(maxLength, PINF<double>));

    // Initialization
    cost[0][0] = ed(series1[0], series2[0]);
    for (long i = 1; i<length1; i++) {
      cost[i][0] = cost[i-1][0]+get_cost(series1[i], series1[i-1], series2[0], c);
    }
    for (long i = 1; i<length2; i++) {
      cost[0][i] = cost[0][i-1]+get_cost(series2[i], series1[0], series2[i-1], c);
    }

    // Main Loop
    for (long i = 1; i<length1; i++) {
      for (long j = 1; j<length2; j++) {
        double d1, d2, d3;
        d1 = cost[i-1][j-1]+ed(series1[i], series2[j]);                        // Diag
        d2 = cost[i-1][j]+get_cost(series1[i], series1[i-1], series2[j], c);     // Prev
        d3 = cost[i][j-1]+get_cost(series2[j], series1[i], series2[j-1], c);     // Top
        cost[i][j] = min(d1, std::min(d2, d3));
      }
    }

    // Output
    return cost[length1-1][length2-1];
  }

  /// Reference multivariate series
  double msm_matrix(const vector<double>& series1, const vector<double>& series2, size_t dim, double c) {
    // Length of the series depends on the actual size of the data and the dimension
    const long length1 = (long) series1.size()/(long) dim;
    const long length2 = (long) series2.size()/(long) dim;

    // Check lengths. Be explicit in the conditions.
    if (length1==0 && length2==0) { return 0; }
    if (length1==0 && length2!=0) { return PINF<double>; }
    if (length1!=0 && length2==0) { return PINF<double>; }

    const long maxLength = max(length1, length2);
    vector<std::vector<double>> cost_matrix(maxLength, std::vector<double>(maxLength, PINF<double>));

    // Initialization
    cost_matrix[0][0] = edN(series1, 0, series2, 0, dim);
    for (long i = 1; i<length1; i++) {
      cost_matrix[i][0] = cost_matrix[i-1][0]+msm_cost(series1, i, i-1, series2, 0, dim, c);
    }
    for (long i = 1; i<length2; i++) {
      cost_matrix[0][i] = cost_matrix[0][i-1]+msm_cost(series2, i, i-1, series1, 0, dim, c);
    }

    // Main Loop
    for (long i = 1; i<length1; i++) {
      for (long j = 1; j<length2; j++) {
        double d1, d2, d3;
        d1 = cost_matrix[i-1][j-1]+edN(series1, i, series2, j, dim);            // Diag
        d2 = cost_matrix[i-1][j]+msm_cost(series1, i, i-1, series2, j, dim, c);   // Prev
        d3 = cost_matrix[i][j-1]+msm_cost(series2, j, j-1, series1, i, dim, c);   // Top
        cost_matrix[i][j] = min(d1, std::min(d2, d3));
      }
    }

    // Output
    return cost_matrix[length1-1][length2-1];
  }
}


// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
TEST_CASE("Multivariate Dependent MSM Fixed length", "[msm][multivariate]") {
  // Setup univariate with fixed length
  Mocker mocker;
  mocker._dim = ndim;
  const auto& msm_costs = mocker.msm_costs;
  const auto fset = mocker.vec_randvec(nbitems);

  SECTION("MSM(s,s) == 0") {
    for (const auto& s: fset) {
      for (auto c: msm_costs) {
        const double dtw_ref_v = msm_matrix(s, s, ndim, c);
        REQUIRE(dtw_ref_v==0);

        const auto dtw_v = msm<double>(s, s, ndim, c);
        REQUIRE(dtw_v==0);
      }
    }
  }

  SECTION("MSM(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i+1];
      for (auto c: msm_costs) {
        // Ensure that things work in the univariate code
        // Note that we recoded the univariate with the Euclidean distance instead of using the L1 norm
        {
          const double dtw_ref_v = msm_matrix(s1, s2, 1, c);
          const double dtw_ref_univ = msm_matrix_uni(s1, s2, c);
          REQUIRE(dtw_ref_v==dtw_ref_univ);
          const double dtw_tempo_v = msm<double>(s1, s2, (size_t) 1, c);
          REQUIRE(dtw_ref_v==dtw_tempo_v);
        }

        // Ok, test the multivariate
        {
          const double dtw_ref_v = msm_matrix(s1, s2, ndim, c);
          const double dtw_tempo_v = msm<double>(s1, s2, ndim, c);
          REQUIRE(dtw_ref_v==dtw_tempo_v);
        }
      }
    }
  }

  SECTION("NN1 MSM") {
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

        for (auto c: msm_costs) {

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const double v_ref = msm_matrix(s1, s2, ndim, c);
          if (v_ref<bsf_ref) {
            idx_ref = j;
            bsf_ref = v_ref;
          }

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v = msm<double>(s1, s2, ndim, c);
          if (v<bsf) {
            idx = j;
            bsf = v;
          }

          REQUIRE(idx_ref==idx);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v_eap = msm<double>(s1, s2, ndim, c, bsf_eap);
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



TEST_CASE("Multivariate Dependent MSM Variable length", "[msm][multivariate]") {
  // Setup univariate with fixed length
  Mocker mocker;
  mocker._dim = ndim;
  const auto& msm_costs = mocker.msm_costs;
  const auto fset = mocker.vec_rs_randvec(nbitems);

  SECTION("MSM(s,s) == 0") {
    for (const auto& s: fset) {
      for (auto c: msm_costs) {
        const double dtw_ref_v = msm_matrix(s, s, ndim, c);
        REQUIRE(dtw_ref_v==0);

        const auto dtw_v = msm<double>(s, s, ndim, c);
        REQUIRE(dtw_v==0);
      }
    }
  }

  SECTION("MSM(s1, s2)") {
    for (size_t i = 0; i<nbitems-1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i+1];
      for (auto c: msm_costs) {
        // Ensure that things work in the univariate code
        // Note that we recoded the univariate with the Euclidean distance instead of using the L1 norm
        {
          const double dtw_ref_v = msm_matrix(s1, s2, 1, c);
          const double dtw_ref_univ = msm_matrix_uni(s1, s2, c);
          REQUIRE(dtw_ref_v==dtw_ref_univ);
          const double dtw_tempo_v = msm<double>(s1, s2, (size_t) 1, c);
          REQUIRE(dtw_ref_v==dtw_tempo_v);
        }

        // Ok, test the multivariate
        {
          const double dtw_ref_v = msm_matrix(s1, s2, ndim, c);
          const double dtw_tempo_v = msm<double>(s1, s2, ndim, c);
          REQUIRE(dtw_ref_v==dtw_tempo_v);
        }
      }
    }
  }

  SECTION("NN1 MSM") {
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
        for (auto c: msm_costs) {

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const double v_ref = msm_matrix(s1, s2, ndim, c);
          if (v_ref<bsf_ref) {
            idx_ref = j;
            bsf_ref = v_ref;
          }

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v = msm<double>(s1, s2, ndim, c);
          if (v<bsf) {
            idx = j;
            bsf = v;
          }

          REQUIRE(idx_ref==idx);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v_eap = msm<double>(s1, s2, ndim, c, bsf_eap);
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