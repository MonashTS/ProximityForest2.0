#pragma once

#include <algorithm>
#include <concepts>

namespace tempo::transform::core::univariate {

    class Hydra {

        // --- From constructor
        DTS const &train_dataset;
        DatasetHeader const &train_header;

        const size_t nb_candidates;
        const size_t nb_trees;

        const std::string str = "pf2";

        const std::string tr_default = "default";
        const std::string tr_d1 = "derivative1";

        const std::vector<std::string> transforms{tr_default, tr_d1};

        const std::vector<F> exponents{0.5, 1, 2};

        shared_ptr<MDTS> train_map = make_shared<MDTS>();
        shared_ptr<MDTS> test_map = make_shared<MDTS>();

        std::shared_ptr<tsc::Forest> forest;
        tsc::TreeData tdata;
        tsc::TreeState &tstate;

}