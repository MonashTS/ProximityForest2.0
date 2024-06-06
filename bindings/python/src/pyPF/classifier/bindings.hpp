#include "pybind_dtw.hpp"
#include "pybind_erp.hpp"
#include "pybind_lcss.hpp"
#include "pybind_msm.hpp"
#include "pybind_squaredED.hpp"
#include "pybind_twe.hpp"

namespace pyPF::classifier {

    /// Add a submodule named "distances" into "m", and register the various (univariate) distances.
    inline void submod_classifier(py::module& m){
        auto sm = m.def_submodule("classifier");
        init_dtw(sm);
        init_erp(sm);
        init_lcss(sm);
        init_msm(sm);
        init_squaredED(sm);
        init_twe(sm);
    }

}
