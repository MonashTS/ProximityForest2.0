#pragma once

#include <libtempo/classifier/proximity_forest/ipf.hpp>
#include <libtempo/tseries/dataset.hpp>
#include <libtempo/concepts.hpp>
#include <libtempo/distance/direct.hpp>
#include <libtempo/distance/dtw.hpp>
#include <libtempo/distance/cdtw.hpp>

#include <random>
#include <utility>
#include <vector>
#include <iostream>
#include <functional>

namespace libtempo::classifier::pf {

  template<Label L, typename Strain, typename Stest>
  requires has_prng<Strain>
  struct SG_chooser : public IPF_NodeGenerator<L, Strain, Stest> {
    using Result = typename IPF_NodeGenerator<L, Strain, Stest>::Result;

    using SGVec_t = std::vector<std::shared_ptr<IPF_NodeGenerator<L, Strain, Stest>>>;

    SGVec_t sgvec;
    size_t nb_candidates;

    SG_chooser(SGVec_t&& sgvec, size_t nb_candidates) :
      sgvec(std::move(sgvec)),
      nb_candidates(nb_candidates) {}

    /** Implementation fo the generate function
     * Randomly generate 'nb_candidates', evaluate them, keep the best (the lowest score is best)
     */
    Result generate(Strain& state, const std::vector<ByClassMap<L>>& bcmvec) const override {
      // Access the pseudo random number generator: state must inherit from PRNG_mt64
      auto& prng = state.prng;
      Result best_result{};
      double best_score = utils::PINF<double>;
      for (size_t i = 0; i<nb_candidates; ++i) {
        const auto idx = std::uniform_int_distribution<size_t>(0, sgvec.size() - 1)(*prng);
        Result result = sgvec[idx]->generate(state, bcmvec);
        double score = weighted_gini_impurity(result);
        if (score<best_score) {
          best_score = score;
          best_result = std::move(result);
        }
      }

      return best_result;
    }

  private:

    //// Compute the weighted (ratio of series per branch) gini impurity of a split.
    [[nodiscard]]
    static double weighted_gini_impurity(const Result& split) {
      double wgini{0};
      double total_size{0};
      for (const auto& bcm : split.branch_splits) {
        double g = bcm.gini_impurity();
        // Weighted part: multiply gini score by the total number of item in this branch
        const double bcm_size = bcm.size();
        wgini += bcm_size*g;
        // Accumulate total size for final division
        total_size += bcm_size;
      }
      // Finish weighted computation by scaling to [0, 1]
      assert(total_size!=0);
      return wgini/total_size;
    }
  };


  /** Leaf generator, stopping at pure node */
  template<Float F, Label L, typename Strain, typename Stest>
  struct SGLeaf_PureNode : public IPF_LeafGenerator<L, Strain, Stest> {
    // Type shorthands
    using Result = typename IPF_LeafGenerator<L, Strain, Stest>::Result;

    /// leaf splitter
    struct PureNode : public IPF_LeafSplitter<L, Stest> {
      std::vector<double> proba;

      PureNode(const std::map<L, size_t>& label_to_index, L label) : proba(label_to_index.size(), 0.0) {
        proba[label_to_index.at(label)] = 1.0;
      }

      std::vector<double> predict_proba(Stest& /* state */ , size_t /* test_index */) const override {
        return proba;
      }
    };

    /// Override interface ISplitterGenerator
    Result generate(Strain& state, const std::vector<ByClassMap<L>>& bcmvec) const override {
      const auto& bcm = bcmvec.back();
      // Generate leaf on pure node
      if (bcm.nb_classes()==1) {
        const auto& header = state.get_header();
        std::string label = bcm.begin()->first;
        return { Result{ResLeaf<L, Stest>{.splitter = std::make_unique<PureNode>(header.label_to_index(), label)}} };
      }
        // Else, return the empty option
      else { return {}; }
    }
  };


}