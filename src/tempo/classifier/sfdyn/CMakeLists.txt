target_sources(libtempo
        PUBLIC
        stree.hpp
        # --- --- --- Leaf
        splitter/leaf/pure_leaf.hpp
        # --- --- --- Node splitter
        splitter/node/meta/chooser.hpp
        # --- NN1 Distance
        splitter/node/nn1/node_nn1dist.hpp
        splitter/node/nn1/nodegen_nn1dist.hpp
        splitter/node/nn1/nn1_directa.hpp
        splitter/node/nn1/MPGenerator.hpp
)