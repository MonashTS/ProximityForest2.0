#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/dts.hpp>
#include <tempo/classifier/utils.hpp>
#include <tempo/classifier/sfdyn/stree.hpp>
#include <utility>

namespace tempo::classifier::sf::leaf {

  /// Pure leaf splitter
  struct SplitterLeaf_Pure : public i_SplitterLeaf {

    // --- --- --- Fields
    /// Pure leaf result is computed at train time
    classifier::Result1 result;

    // --- --- --- Constructor / Destructors
    /// Construction with already built result
    explicit SplitterLeaf_Pure(classifier::Result1&& r) : result(std::move(r)) {}

    // --- --- --- Methods
    /// Simply return a copy of the stored result
    classifier::Result1 predict(TreeState& /* state */, TreeData const& /* data */, size_t /* index */) override {
      return result;
    }

  };

  /// Pure leaf generator: stop when only one class reaches the node
  struct GenLeaf_Pure : public i_GenLeaf {

    // --- --- --- types

    std::shared_ptr<i_GetData<DatasetHeader>> get_train_header;


    // --- --- --- Constructors/Destructors

    explicit GenLeaf_Pure(std::shared_ptr<i_GetData<DatasetHeader>> get_train_header) :
      get_train_header(std::move(get_train_header)) {}

    // --- --- --- Methods

    i_GenLeaf::Result generate(TreeState& /* state */, TreeData const& data, ByClassMap const& bcm) override {
      // Generate leaf on pure node:
      // Vector of probabilities at 0 except for the position matching the encoded label
      if (bcm.nb_classes()==1) {
        size_t cardinality = get_train_header->at(data).nb_classes();
        EL elabel = *bcm.classes().begin();  // Get the encoded label
        return {
          std::make_unique<SplitterLeaf_Pure>(
            classifier::Result1::make_probabilities_one(cardinality, elabel, 1.0)
          )
        };
      } else { return {}; } // Else, return the empty option
    }

  };

} // End of namespace tempo::classifier::SForest::leaf