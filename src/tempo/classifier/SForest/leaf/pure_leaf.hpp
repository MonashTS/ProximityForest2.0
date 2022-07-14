#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/dts.hpp>
#include "tempo/classifier/SForest/stree.hpp"

namespace tempo::classifier::SForest::leaf {

  /// Pure leaf generator: stop when only one class reaches the node
  template<TreeState TrainS, TrainData TrainD, typename TestS, typename TestD>
  struct PureLeaf_Gen : public LeafSplitterGen_i<TrainS, TrainD, TestS, TestD> {

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Internal class for the test time splitter
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    struct PureLeaf : public LeafSplitter_i<TestS, TestD> {
      using R = typename LeafSplitter_i<TestS, TestD>::R;

      /// Pure leaf result is computed at train time
      classifier::Result result;

      /// Construction with already built result
      explicit PureLeaf(classifier::Result&& r) : result(std::move(r)) {}

      /// Simply return a copy of the stored result
      R predict(std::unique_ptr<TestS> state, const TestD& /* data */, size_t /* index */) override {
        return R{std::move(state), result};
      }
    };

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Splitter generator code at train time
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    using R = typename LeafSplitterGen_i<TrainS, TrainD, TestS, TestD>::R;

    R generate(std::unique_ptr<TrainS> state, const TrainD& data, const ByClassMap& bcm) override {
      if (bcm.nb_classes()==1) { // Generate leaf on pure node
        size_t cardinality = data.get_train_header().nb_classes();
        EL elabel = *bcm.classes().begin();   // Get the encoded label
        auto weight = (double)bcm.size();     // Get the "weight", i.e. the cardinality of the bcm
        auto leafptr = std::make_unique<PureLeaf>(
          classifier::Result::make_probabilities_one(cardinality, elabel, weight)
        );
        return R{std::move(state), {std::move(leafptr)}};
      } else { return R{std::move(state), {}}; } // Else, return the empty option
    }

  };

} // End of namespace tempo::classifier::SForest::leaf