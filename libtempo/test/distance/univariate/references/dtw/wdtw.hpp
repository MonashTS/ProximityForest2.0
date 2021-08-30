#pragma once

#include <vector>
#include <cassert>
#include <cmath>


// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Reference implementation, with square euclidean distance
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
namespace reference {

    /// Reference implementation on a matrix
    double wdtw_matrix(const double *series1, size_t length1_, const double *series2, size_t length2_, const double *weights);

    /// Wrapper for vector
    inline double wdtw_matrix(
            const std::vector<double> &series1,
            const std::vector<double> &series2,
            const std::vector<double> &weights
            ) {
        assert(series1.size() <= weights.size());
        assert(series2.size() <= weights.size());
        return wdtw_matrix(series1.data(), series1.size(), series2.data(), series2.size(), weights.data());
    }

} // End of namespace references
