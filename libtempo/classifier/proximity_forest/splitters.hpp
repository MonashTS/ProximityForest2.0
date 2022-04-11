#pragma once

#include <libtempo/classifier/proximity_forest/ipf.hpp>
#include <libtempo/tseries/dataset.hpp>
#include <libtempo/concepts.hpp>

#include <random>
#include <utility>
#include <vector>
#include <iostream>
#include <functional>

namespace libtempo::classifier::pf {

  //// Compute the weighted (ratio of series per branch) gini impurity of a split.
  template<Label L, typename Strain, typename Stest>
  [[nodiscard]]
  static double weighted_gini_impurity(const typename IPF_NodeGenerator<L, Strain, Stest>::Result& result) {
    double wgini{0};
    double total_size{0};
    for (const auto& bcm : result.branch_splits) {
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

  template<Label L, typename Strain, typename Stest> requires has_prng<Strain>
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
      Result best_result{};
      double best_score = utils::PINF<double>;
      for (size_t i = 0; i<nb_candidates; ++i) {
        Result result = utils::pick_one(sgvec, *state.prng)->generate(state, bcmvec);
        double score = weighted_gini_impurity<L, Strain, Stest>(result);
        if (score<best_score) {
          best_score = score;
          best_result = std::move(result);
        }
      }

      return best_result;
    }

  };

  template<Label L, typename Strain, typename Stest> requires has_prng<Strain>
  struct SG_try_all : public IPF_NodeGenerator<L, Strain, Stest> {
    using Result = typename IPF_NodeGenerator<L, Strain, Stest>::Result;

    using SGVec_t = std::vector<std::shared_ptr<IPF_NodeGenerator<L, Strain, Stest>>>;

    SGVec_t sgvec;

    SG_try_all(SGVec_t&& sgvec) :
      sgvec(std::move(sgvec)) {}

    /** Implementation fo the generate function
     * Randomly generate 'nb_candidates', evaluate them, keep the best (the lowest score is best)
     */
    Result generate(Strain& state, const std::vector<ByClassMap<L>>& bcmvec) const override {
      Result best_result{};
      double best_score = utils::PINF<double>;
      for (size_t i = 0; i<sgvec.size(); ++i) {
        Result result = sgvec[i]->generate(state, bcmvec);
        double score = weighted_gini_impurity<L, Strain, Stest>(result);
        if (score<best_score) {
          best_score = score;
          best_result = std::move(result);
        }
      }

      return best_result;
    }

  };

  /** Leaf generator, stopping at pure node */
  template<Float F, Label L, typename Strain, typename Stest>
  struct SGLeaf_PureNode : public IPF_LeafGenerator<L, Strain, Stest> {
    // Type shorthands
    using Result = typename IPF_LeafGenerator<L, Strain, Stest>::Result;

    /// leaf splitter
    struct PureNode : public IPF_LeafSplitter<L, Stest> {

      double weight;
      std::vector<double> proba;

      PureNode(double weight, const std::map<L, size_t>& label_to_index, L label) :
        weight(weight),
        proba(label_to_index.size(), 0.0) {
        proba[label_to_index.at(label)] = 1.0;
      }

      std::tuple<double, std::vector<double>> predict_proba(Stest& /* state */ ,
                                                            size_t /* test_index */) const override {
        return {weight, proba};
      }
    };

    /// Override interface ISplitterGenerator
    Result generate(Strain& state, const std::vector<ByClassMap<L>>& bcmvec) const override {
      const auto& bcm = bcmvec.back();
      // Generate leaf on pure node
      if (bcm.nb_classes()==1) {
        const auto& header = state.get_header();
        std::string label = bcm.begin()->first;
        double weight = bcm.size();
        return {
          Result{ResLeaf<L, Strain, Stest>{.splitter = std::make_unique<PureNode>(weight, header.label_to_index(), label)}}};
      }
        // Else, return the empty option
      else { return {}; }
    }
  };

}