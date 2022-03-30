#pragma once

#include <libtempo/concepts.hpp>
#include <libtempo/tseries/dataset.hpp>

#include <functional>
#include <variant>

namespace libtempo::classifier::pf {

  /** Interface with functions to be override for the train state */
  template<Label L>
  struct IStrain {


    virtual ~IStrain() = default;

  };

  /** Interface for splitters at test time
   * @tparam L      Label type
   * @tparam Stest  State type at test time - note that this is a mutable reference
   */
  template<Label L, typename Stest>
  struct ISplitter {

    virtual size_t classify(Stest& state, size_t index) = 0;

    virtual ~ISplitter() = default;
  };

  /** Splitter generator
   * Returns a SplitterGenerator::Result
   * A Splitter Generator must return a splitter for the node, and the split of the train data reaching that node.
   * To do so, it is provided with the train state, the ByClasMap at the node.
   * All the logic about splitters generation/selection goes into the generator.
   * Any source of randomness should come from the states
   * @tparam L       Label type
   * @tparam Strain  State type at train time
   * @tparam Stest   State type at test time
   */
  template<Label L, typename Strain, typename Stest>
  struct ISplitterGenerator {

    /// Result of a splitter generator
    struct Result {

      /// The actual split: the size of the vector tells us the number of branches
      /// Note: if a branch is required, but no data reaches it, this branch will become a pure node.
      std::vector<ByClassMap<L>> branch_splits;

      /// Actual splitter to use
      std::unique_ptr<ISplitter<L, Stest>> splitter;
    };

    /** Generate a new splitter from a training state and the ByClassMap at the node.
     * @param state  Training state - mutable reference!
     * @param bcm    BCM at the node
     * @return  ISplitterGenerator::Result with the splitter and the associated split of the train data
     */
    virtual std::unique_ptr<Result> generate(Strain& state, const ByClassMap <L>& bcm) const = 0;

    virtual ~ISplitterGenerator() = default;
  };

}
























//   /** Split evaluator - lower score represent a more precise split.
//  * When comparing splits, the split with the lowest score will be chosen.
//  * If several splits have the same lowest score, then the first encountered one is picked.
//  * */
// template<typename Fun, typename L>
// concept SplitEvaluator = Label<L>&&requires(Fun&& fun, Split<L>&& split){
//   { // fun(split)->double
//   std::invoke(std::forward<Fun>(fun), std::forward<Split<L>>(split))
//   } -> std::convertible_to<double>;
// };






// /** Split evaluator: compute the weighted (ratio of series per branch) gini impurity of a split. */
// template<Label L>
// [[nodiscard]] static double weighted_gini_impurity(const Split<L>& split) {
//   double wgini{0};
//   double split_size{0};  // Accumulator, total number of series received at this node.
//   // For each branch (i.e. assigned class c), get the actual mapping class->series (bcm)
//   // and compute the gini impurity of the branch
//   for (const auto&[c, bcm_vec] : split) {
//     const auto[bcm, vec] = bcm_vec;
//     double g = bcm.gini_impurity();
//     // Weighted part: the total number of item in this branch is the lenght of the vector of index
//     // Accumulate the total number of item in the split (sum the branches),
//     // and weight current impurity by size of the branch
//     const double bcm_size = vec.size();
//     split_size += bcm_size;
//     wgini += bcm_size*g;
//   }
//   // Finish weighted computation by scaling to [0, 1]
//   return wgini/split_size;
// }


//   /** Splitter
//    * A Splitter is analogous to a classifier than can be trained on a dataset.
//    * The splitter must be able to access the "State", which must contains the train data (at train time)
//    * and the test data (at test time).
//    * We use two different function for classification, as the process at train time might be different from
//    * the process at test time.
//    * If a splitter maintains an internal state,
//    * it must do so through a share_ptr shared in the closure of the functions.
//    * The state can be safely updated without synchronisation.
//    * @tparam L      Label type
//    * @tparam State  State type
//    * @tparam PRNG   Pseudo Random Number Generator type
//    */
//   template<Label L, State<L> S, typename PRNG>
//   struct Splitter_ {
//     std::function<void(std::shared_ptr<S>& state, const IndexSet& is, const ByClassMap<L>& bcm, PRNG& prng)> train;
//     std::function<L(std::shared_ptr<S>& state, size_t index, PRNG& prng)> classify_train;
//     std::function<L(std::shared_ptr<S>& state, size_t index, PRNG& prng)> classify_test;
//   };
