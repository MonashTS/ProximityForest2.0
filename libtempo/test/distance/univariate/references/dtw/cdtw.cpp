#include "cdtw.hpp"

#include "../mock/mockseries.hpp"

namespace reference {

  using namespace std;
  using namespace mock;
  using namespace libtempo::utils;

  /// Naive DTW with a window. Reference code.
  double cdtw_matrix(
    const double *series1, ssize_t length1,
    const double *series2, ssize_t length2,
    size_t w){

    // Check lengths. Be explicit in the conditions
    if (length1 == 0 && length2 == 0) { return 0; }
    if (length1 == 0 && length2 != 0) { return PINF<double>; }
    if (length1 != 0 && length2 == 0) { return PINF<double>; }

    // Allocate the working space: full matrix + space for borders (first column / first line)
    size_t msize = max(length1, length2) + 1;
    vector<std::vector<double>> matrix(msize, std::vector<double>(msize, PINF<double>));

    // Initialisation (all the matrix is initialised at +INF)
    matrix[0][0] = 0;

    // For each line
    // Note: series1 and series2 are 0-indexed while the matrix is 1-indexed (0 being the borders)
    //       hence, we have i-1 and j-1 when accessing series1 and series2
    for (long i = 1; i <= length1; i++) {
      auto series1_i = series1[i - 1];
      long jStart = max<long>(1, i - w);
      long jStop = min<long>(i + w, length2);
      for (long j = jStart; j <= jStop; j++) {
        double prev = matrix[i][j - 1];
        double diag = matrix[i - 1][j - 1];
        double top = matrix[i - 1][j];
        matrix[i][j] = min(prev, std::min(diag, top)) + square_dist(series1_i, series2[j - 1]);
      }
    }

    return matrix[length1][length2];
  }

} // End of namespace references