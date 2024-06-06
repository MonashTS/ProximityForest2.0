#include <pybind11/pybind11.h>

#include "tseries/bindings.hpp"
#include "classifier/pybind_pf2.hpp"

PYBIND11_MODULE(pyPF, m) {

    // --- --- --- Add Time Series
    pytempo::submod_tseries(m);

    // --- --- --- Add Univariate
    // Create a submodule
    auto mod_classifier = m.def_submodule("classifier");
    // Add classifiers
    pyPF::classifier::submod_classifier(mod_classifier);
}


