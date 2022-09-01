target_sources(libtempo
        PUBLIC
        ../utils.hpp
        stree.hpp
        sforest.hpp
        # Leaf
        leaf/pure_leaf.hpp
        # Meta splitters
        splitter/meta/chooser.hpp
        # NN1 splitters
        splitter/nn1/nn1splitters.hpp
        splitter/nn1/MPGenerator.hpp
        splitter/nn1/sp_basedist.hpp
        splitter/nn1/sp_da.hpp
        splitter/nn1/sp_adtw.hpp
        splitter/nn1/sp_dtw.hpp
        splitter/nn1/sp_wdtw.hpp
        splitter/nn1/sp_erp.hpp
        splitter/nn1/sp_lcss.hpp
        splitter/nn1/sp_msm.hpp
        #splitter/nn1/sp_lorentzian.hpp
        #splitter/nn1/sp_sbd.hpp
        splitter/nn1/sp_twe.hpp
        ### ### ### ### ### ### ### ### ### ### ### ###
        PRIVATE
        # NN1 splitters implementation
        splitter/nn1/implem.cpp
        )