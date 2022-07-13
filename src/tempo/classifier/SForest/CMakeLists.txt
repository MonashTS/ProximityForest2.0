target_sources(libtempo
        PUBLIC
        ../utils.hpp
        sforest.hpp
        # Leaf
        leaf/pure_leaf.hpp
        # Meta splitters
        splitter/meta/chooser.hpp
        # NN1 splitters
        splitter/nn1/nn1splitters.hpp
        splitter/nn1/MPGenerator.hpp
        splitter/nn1/sp_da.hpp
        ### ### ### ### ### ### ### ### ### ### ### ###
        PRIVATE
        # NN1 splitters
        splitter/nn1/sp_da.cpp
        )