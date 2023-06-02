#pragma once

#include <algorithm>
#include <concepts>
#include <cmath>
#include <fftw3.h>

using namespace std;

#define REAL 0
#define IMAG 1

namespace tempo::transform::core::univariate {

    /** Transform a series into its power spectrum (frequency domain)
     * @tparam T            Input series, must be in
     * @param series        Pointer to the series's data
     * @param length        Length of the series
     * @param out           Pointer where to write the derivative. Must be able to store 'length' values.
     * Warning: series and out should not overlap (i.e. no in-place derivation)
     */
    template<std::floating_point F, std::random_access_iterator Input, std::output_iterator<F> Output>
    void freqtransform(Input const &series, size_t length, Output &out) {
        fftw_complex x[length];     // input array
        fftw_complex y[length];     // output array
        // convert the data into complex
        for (size_t i = 0; i < length; i++) {
            x[i][REAL] = series[i];
            x[i][IMAG] = 0;
        }
        fftw_plan plan = fftw_plan_dft_1d(length, x, y, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(plan);

        fftw_destroy_plan(plan);
        for (size_t i = 0; i < length; i++) {
            double real2 = y[i][REAL] * y[i][REAL];
            double imag2 = y[i][IMAG] * y[i][IMAG];
            out[i] = std::sqrt(real2 + imag2);
        }
    }


//    void freqtransform2(double *series, size_t length, double *out) {
//        fftw_complex x[length];     // input array
//        fftw_complex y[length];     // output array
//        // convert the data into complex
//        for (size_t i = 0; i < length; i++) {
//            x[i][REAL] = series[i];
//            x[i][IMAG] = 0;
//        }
//        fftw_plan plan = fftw_plan_dft_1d(length, x, y, FFTW_FORWARD, FFTW_ESTIMATE);
//        fftw_execute(plan);
//
//        fftw_destroy_plan(plan);
//        for (size_t i = 0; i < length; i++) {
//            double real2 = y[i][REAL] * y[i][REAL];
//            double imag2 = y[i][IMAG] * y[i][IMAG];
//            out[i] = std::sqrt(real2 + imag2);
//        }
//    }

}