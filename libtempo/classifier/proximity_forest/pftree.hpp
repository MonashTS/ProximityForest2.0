#pragma once

#include <libtempo/concepts.hpp>
#include <libtempo/tseries/dataset.hpp>
#include <libtempo/utils/utils.hpp>

#include <functional>
#include <map>
#include <string>
#include <variant>

namespace libtempo::classifier::pf {

  /// Concept describing the minimum function required on a State
  template<typename S, typename L>
  concept State = Label<L> && requires(S state, size_t i, size_t j){
    { state.get_label(i) }->std::same_as<std::optional<L>>;
  };

  /** Splitter
   * A Splitter is analogous to a classifier than can be trained on a dataset.
   * The splitter must be able to access the "State", which must contains the train data (at train time)
   * and the test data (at test time).
   * We use two different function for classification, as the process at train time might be different from
   * the process at test time.
   * If a splitter maintains an internal state,
   * it must do so through a share_ptr shared in the closure of the functions.
   * The state can be safely updated without synchronisation.
   * @tparam L      Label type
   * @tparam State  State type
   * @tparam PRNG   Pseudo Random Number Generator type
   */
  template<Label L, State<L> S, typename PRNG>
  struct Splitter {
    std::function<void(std::shared_ptr<S>& state, const IndexSet& is, const ByClassMap<L>& bcm, PRNG& prng)> train;
    std::function<L(std::shared_ptr<S>& state, size_t index, PRNG& prng)> classify_train;
    std::function<L(std::shared_ptr<S>& state, size_t index, PRNG& prng)> classify_test;
  };

  template<Label L, State<L> S, typename PRNG>
  using Splitter_uptr = std::unique_ptr<Splitter<L, S, PRNG>>;

  /** Splitter generator
   * A Splitter generator creates a splitter given a State, the set of indexes at this node, and a source of randomness.
   * Note that splitter generator are shared between multiple threads. If they update some internal state (closure),
   * they must do so in a thread-safe way.
   * The 'state' can be updated without synchronisation (one per thread/tree).
   * @tparam L
   * @tparam S
   * @tparam PRNG
   */
  template<Label L, State<L> S, typename PRNG>
  struct SplitterGenerator {

    std::function<Splitter_uptr<L, S, PRNG>(
      std::shared_ptr<S> state,
      const IndexSet& is,
      const ByClassMap<L>& bcm,
      PRNG& prng
    )> generate;

  };

  template<Label L, State<L> S, typename PRNG>
  struct PFTree {

    /** Type of a splitter pointer in this code */
    using Splitter_ptr = Splitter_uptr<L, S, PRNG>;

    /** Record ratio of subtree */
    using SplitRatios = std::vector<std::tuple<L, double>>;

    /** Mapping between a class (label) to a branch (subnode) */
    using Branches = std::unordered_map<L, std::unique_ptr<PFTree>>;

    /** Pure node: only contains a label. It is a leaf node. */
    struct Leaf { L label; };

    /** Inner node. */
    struct Node {

      /// Splitter at this node
      Splitter_ptr splitter;

      /// Branches at this node
      Branches branches;

      /// Train time information TODO: move in state
      SplitRatios train_split_ratios;
    };

    /** Flag: is it a pure node or not? */
    bool is_pure_node;

    /** Pure xor Inner node */
    std::variant<Leaf, Node> node;

    /** Type of a split.
     *  A mapping    predicted label (i.e. the branch) ->  (map true label-> series index, series index)
     *  Keeping the map true label -> series index allows for easier computation of the gini impurity.
     */
    using Split = std::unordered_map<L, std::tuple<ByClassMap<L>, std::vector<size_t>>>;

    /** Compute the weighted (ratio of series per branch) gini impurity of a split. */
    [[nodiscard]] static double weighted_gini_impurity(const Split& split) {
      double wgini{0};
      double split_size{0};  // Accumulator, total number of series received at this node.
      // For each branch (i.e. assigned class c), get the actual mapping class->series (bcm)
      // and compute the gini impurity of the branch
      for (const auto&[c, bcm_vec]: split) {
        const auto[bcm, vec] = bcm_vec;
        double g = gini_impurity(bcm);
        // Weighted part: the total number of item in this branch is the lenght of the vector of index
        // Accumulate the total number of item in the split (sum the branches),
        // and weight current impurity by size of the branch
        const double bcm_size = vec.size();
        split_size += bcm_size;
        wgini += bcm_size*g;
      }
      // Finish weighted computation by scaling to [0, 1]
      return wgini/split_size;
    }

    /** Make a split, evaluate it, return it with its gini impurity */
    [[nodiscard]] static std::tuple<Splitter_ptr, Split, double> mk_split(
      std::shared_ptr<S>& state,
      const IndexSet& is,
      const ByClassMap<L>& bcm,
      SplitterGenerator<L, S, PRNG>& sg,
      PRNG& prng
    ) {
      // Generate a splitter and train it.
      Splitter_ptr splitter = sg.generate(state, is, bcm, prng);
      splitter->train(state, is, bcm, prng);
      Split split;
      // "train classify" each index in the 'is' subset
      for (const auto& query_idx: is) {
        // Predict the label, and get the actual label
        const L predicted_label = splitter->classify_train(state, query_idx, prng);
        L actual_label = state->get_label(query_idx).value();  // Here, crash if no values...
        // Update the branch of the predicted label, but writing in the actual label (will show us the disagreements)
        auto&[branch_bcm, branch_vec] = split[predicted_label];
        branch_bcm[actual_label].push_back(query_idx);
        branch_vec.push_back(query_idx);
      }
      // Compute the weighted gini of the splitter
      double wg = weighted_gini_impurity(split);
      return {std::move(splitter), std::move(split), wg};
    }

    /** Recursively build a tree */
    [[nodiscard]] static std::unique_ptr<PFTree> make_tree(
      std::shared_ptr<S>& state,
      const IndexSet& is,
      const ByClassMap<L>& bcm,
      size_t nbcandidates,
      SplitterGenerator<L, S, PRNG>& sg,
      PRNG& prng
    ) {

      using namespace std;
      assert(bcm.size()>0);

      // --- --- --- CASE 1 - leaf case: only one class in bcm
      if (bcm.size()==1) {
        return unique_ptr<PFTree>(new PFTree{
          .is_pure_node=true,
          .node=Leaf{bcm.begin()->first}
        });
      }

      // --- --- --- CASE 2 - internal node case
      // Best variables: gini, associated splitter and split (for each branch, the by class map)
      double best_gini = utils::PINF<double>;
      Splitter_ptr best_splitter;
      Split best_split;

      // Generate and evaluate the candidate splitters by computing their weighted gini.
      // Save the best (less impure).
      for (size_t n = 0; n<nbcandidates; ++n) {
        auto[splitter, split, gini] = mk_split(state, is, bcm, sg, prng);
        if (gini<best_gini) {
          best_gini = gini;
          best_splitter = move(splitter);
          best_split = move(split);
        }
      }

      // Now, we have our best candidate. Recursively create the sub trees and then create the node itself.
      // Note: iterate using the incoming 'by_class' map and not the computed 'split' as the split may not contains
      // all incoming classes (i.e. never selected by the splitter)
      unordered_map<L, unique_ptr<PFTree>> sub_trees;
      std::vector<std::tuple<L, double>> split_ratios;
      auto size = (double) is.size();
      for (const auto&[label, _]: bcm) {
        if (best_split.contains(label)) {
          auto[bcm, indexes] = std::move(best_split[label]);
          split_ratios.push_back({label, indexes.size()/size});
          sub_trees[label] = make_tree(state, IndexSet(std::move(indexes)), bcm, nbcandidates, sg, prng);
        } else {
          // Label not showing up at all in the split (never selected by the splitter). Create a leaf.
          sub_trees[label] = unique_ptr<PFTree>(new PFTree{.is_pure_node=true, .node=Leaf{label}});
          split_ratios.push_back({label, 0});
        }
      }

      sort(split_ratios.begin(), split_ratios.end(), [](const auto& a, const auto& b) -> bool {
        return get<1>(a)>get<1>(b);
      });

      return unique_ptr<PFTree>(new PFTree{.is_pure_node=false, .node={
        Node{.splitter=std::move(best_splitter),
          .branches=std::move(sub_trees),
          .train_split_ratios=std::move(split_ratios)}
      }});

    }


    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // --- Classifier interface
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /** Classifier interface */
    struct Classifier {
    private:
      // Fields
      const PFTree& pt;
      PRNG& prng;

      // Main classification function (stateless)
      [[nodiscard]] static L classify(const PFTree& pt, std::shared_ptr<S>& state, size_t idx, PRNG& prng) {
        if (pt.is_pure_node) {
          // Leaf: returns the label
          return std::get<0>(pt.node).label;
        } else {
          const Node& n = std::get<1>(pt.node);
          L res = n.splitter->classify_test(state, idx, prng);
          const auto& sub = n.branches.at(res);    // Use at (is const, when [ ] is not)
          return classify(*sub, state, idx, prng);
        }
      }

    public:

      Classifier(const PFTree& pt, PRNG& prng)
        :pt(pt), prng(prng) { }

      [[nodiscard]] L classify(std::shared_ptr<S>& state, size_t index) {
        return classify(pt, state, index, prng);
      }
    };// End of Classifier

    Classifier get_classifier(PRNG& prng) { return Classifier(*this, prng); }


  };


}