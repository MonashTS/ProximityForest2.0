#pragma once

#include <libtempo/concepts.hpp>
#include <libtempo/tseries/dataset.hpp>
#include <libtempo/utils/utils.hpp>
#include <libtempo/classifier/proximity_forest/ipf.hpp>

#include <functional>
#include <map>
#include <string>
#include <variant>

namespace libtempo::classifier::pf {

  /** A Proximity Forest Tree
   * Such a tree is made of nodes, either leaf (pure node) or inner node.
   * The top node represents the full tree.
   * @tparam L          Label type
   * @tparam Stest      State at test type
   */
  template<Label L, typename Stest>
  struct PFTree {

    /** Pure node: only contains a label. It is a leaf node. */
    struct Leaf { L label; };

    /** Inner node. */
    struct Node {

      /// Selected splitter at this node
      std::unique_ptr<ISplitter<L, Stest>> splitter;

      /** Branches at this node. The splitters must return a branch index [0, nb_branches[ per instance */
      std::vector<std::unique_ptr<PFTree>> branches;
    };

    /** Leaf Node xor Inner Node */
    std::variant<Leaf, Node> node;

    /** Recursively build a tree
     * @tparam Strain   Type of the state at train time
     * @param state     State at train time - note that this is a mutable reference!
     * @param bcm       Initial ByClassMap over the full train set
     * @param sg        Splitter Generator
     * @return
     */
    template<typename Strain>
    [[nodiscard]] static std::unique_ptr<PFTree<L, Stest>> make_tree(
      Strain& state,
      const ByClassMap<L>& bcm,
      const ISplitterGenerator<L, Strain, Stest>& sg
    ) {
      // Ensure that we have at least one class reaching this node!
      assert(!bcm.empty());

      // Enter make tree
      auto& state_ = static_cast<IStrain<L>&>(state);
      state_.on_make_tree(bcm);

      std::unique_ptr<PFTree<L, Stest>> ret;

      if (bcm.nb_classes()==1) {
        // --- --- --- CASE 1 - leaf case: only one class in bcm
        L label = bcm.begin()->first;
        state_.on_make_leaf(label);
        ret = std::unique_ptr<PFTree<L, Stest>>(new PFTree<L, Stest>{.node=Leaf{label}});
      } else {
        // --- --- --- CASE 2 - internal node case
        // Generate the splitter
        auto result = sg.generate(state, bcm);
        std::vector<std::unique_ptr<PFTree<L, Stest>>> subbranches;
        // Recursively create the subtrees.
        for (size_t i = 0; i<result->branch_splits.size(); ++i) {
          ByClassMap<L> branch_bcm = std::move(result->branch_splits[i]);
          subbranches.push_back(make_tree(state, branch_bcm, sg));
        }
        // Create the node itself.
        ret = std::unique_ptr<PFTree<L, Stest>>(new PFTree<L, Stest>{
                                                  .node={Node{
                                                    .splitter=std::move(result->splitter),
                                                    .branches=std::move(subbranches)
                                                  }}
                                                }
        );
      }

      // Exit make tree - always executed
      state.exit_make_tree();
      return ret;

    } // End of make_tree


    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // --- Helpers
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /** Helper checking the above variant */
    [[nodiscard]]
    bool is_pure_node() const { return node.index()==0; }


    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // --- Classifier
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /** Classifier interface */
    class Classifier {

      const PFTree& pt;

      // Main classification function (stateless)
      [[nodiscard]] static L classify(const PFTree& pt, Stest& state, size_t idx) {
        switch (pt.node.index()) {
        case 0: { return std::get<0>(pt.node).label; }
        case 1: {
          const Node& n = std::get<1>(pt.node);
          L res = n.splitter->classify(state, idx);
          const auto& sub = n.branches.at(res);    // Use at because it is 'const', when [ ] is not
          return classify(*sub, state, idx);
        }
        default: utils::should_not_happen();
        }
      }

      explicit Classifier(const PFTree& pt) : pt(pt) {}

    public:

      [[nodiscard]] L classify(std::shared_ptr<Stest>& state, size_t index) { return classify(pt, state, index); }

    };// End of Classifier

    [[nodiscard]]
    Classifier get_classifier() { return Classifier(*this); }
  };

}