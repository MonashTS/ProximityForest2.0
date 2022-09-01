target_sources(libtempo
        PUBLIC
        # --- --- --- Interfaces
        treedata.hpp
        treestate.hpp
        splitter_interface.hpp
        # --- --- --- Tree
        tree.hpp
        # --- --- --- Splitter: leaf
        sleaf/pure_leaf.hpp
        # --- --- --- Splitter: Node
        # --- Meta nodes
        snode/meta/chooser.hpp
        # --- NN1 Distance
        snode/nn1/node_nn1dist.hpp
        snode/nn1/nodegen_nn1dist.hpp
        snode/nn1/nn1_directa.hpp
        snode/nn1/nn1_dtwfull.hpp
        snode/nn1/MPGenerator.hpp
        # --- --- ---
        # --- --- ---
        # --- --- ---
        PRIVATE
        tree.cpp
        treestate.cpp
)