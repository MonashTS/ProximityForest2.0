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
    using LeafSplitter = std::unique_ptr<IPF_LeafSplitter<L, Stest>>;
    using InnerNodeSplitter = std::unique_ptr<IPF_NodeSplitter<L, Stest>>;
    using Branches = std::vector<std::unique_ptr<PFTree<L, Stest>>>;

    struct InnerNode {
      InnerNodeSplitter splitter;
      Branches branches;
    };

    /** Leaf Node xor Inner Node */
    std::variant<LeafSplitter, InnerNode> node;

    /** Recursively build a tree
     * @tparam Strain   Type of the state at train time
     * @param strain     State at train time - note that this is a mutable reference!
     * @param bcm       Initial ByClassMap over the full train set
     * @param sg        Splitter Generator
     * @return
     */
    template<typename Strain> requires std::derived_from<Strain, IStrain<L, Strain, Stest>>
    [[nodiscard]] static std::unique_ptr<PFTree<L, Stest>> make_node(
      Strain& strain,
      std::vector<ByClassMap<L>> bcmvec,
      const IPF_NodeGenerator<L, Strain, Stest>& sg
    ) {
      using Result = typename IPF_NodeGenerator<L, Strain, Stest>::Result;
      // Ensure that we have at least one class reaching this node!
      // Note: there may be no data point associated to the class.
      const auto bcm = bcmvec.back();
      assert(bcm.nb_classes()>0);
      // Final return 'ret' variable
      std::unique_ptr<PFTree<L, Stest>> ret;
      // Call the generator and analyse the result
      Result result = sg.generate(strain, bcmvec);
      switch (result.index()) {
      case 0: { // Leaf case - stop the recursion, build a leaf node
        ResLeaf<L, Stest> leaf = std::get<0>(std::move(result));
        // Train state callback
        strain.on_make_leaf(leaf);
        // Build the leaf node with the splitter
        ret = std::unique_ptr<PFTree<L, Stest>>(new PFTree<L, Stest>{.node=std::move(leaf.splitter)});
        break;
      }
      case 1: { // Inner node case: recursion per branch
        ResNode<L, Stest> inner_node = std::get<1>(std::move(result));
        // Train state callback
        strain.on_make_branches(inner_node);
        // Build subbranches, then we'll build the current node
        Branches subbranches;
        const size_t nbbranches = inner_node.branch_splits.split.size();
        for (size_t idx = 0; idx<nbbranches; ++idx) {
          // Clone state, push bcm
          bcmvec.template emplace_back(std::move(inner_node.branch_splits.split[idx]));
          Strain sub_state = strain.clone(idx);
          // Sub branch
          subbranches.push_back(make_node(sub_state, bcmvec, sg));
          // Merge state, pop bcm
          strain.merge(std::move(sub_state));
          bcmvec.pop_back();
        }
        // Now that we have all the subbranches, build the current node
        ret = std::unique_ptr<PFTree<L, Stest>>(
          new PFTree<L, Stest>{.node= InnerNode{
            .splitter = std::move(inner_node.splitter),
            .branches = std::move(subbranches)
          }}
        );
        break;
      }
      default: utils::should_not_happen();
      }
      return ret;
    } // End of make_node


    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // --- Classifier
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /** Classifier interface */
    class Classifier {

      const PFTree& pt;

      // Main classification function (stateless)
      [[nodiscard]] static std::vector<double> classify(const PFTree& pt, Stest& state, size_t query_idx) {
        switch (pt.node.index()) {
        case 0: { return std::get<0>(pt.node)->predict_proba(state, query_idx); }
        case 1: {
          const InnerNode& n = std::get<1>(pt.node);
          size_t branch_idx = n.splitter->get_branch_index(state, query_idx);
          const auto& sub = n.branches.at(branch_idx);    // Use at because it is 'const', when [ ] is not
          return classify(*sub, state, query_idx);
        }
        default: utils::should_not_happen();
        }
      }

    public:

      explicit Classifier(const PFTree& pt) : pt(pt) {}

      [[nodiscard]] std::vector<double> classify(Stest& state, size_t index) { return classify(pt, state, index); }

    };// End of Classifier

    [[nodiscard]]
    Classifier get_classifier() { return Classifier(*this); }
  };

}