#pragma once

#include <tempo/predef.hpp>
#include <tempo/tseries/dataset.hpp>

namespace tempo::classifier {

  struct Result {

    /// Resulting (nbclass X nbtest_exemplar) matrix of probabilities.
    /// Each column represents the result of one test exemplar, in the order
    /// found in the test set. Inside a column, each row represents the probability of the class matching the index,
    /// This matching is done with the label encoder provided by the test dataset header.
    /// In order to properly build this encoder, it must be built with the encoder obtained at train time.
    arma::mat probabilities;

    /// Row vector of length nbtest_exemplar,
    /// indicating the "weights" (can be used to indicate confidence)
    /// associated to each probability vector from the probabilities matrix
    arma::rowvec weights;

    /// row vector of length nbtest_exemplar
    /// indicating for each exemplar the index of the actual class in the label encoder.
    /// Use -1 to indicate unknown label
    arma::Row<int> true_labels;

  };

  /// Record results in 'out', which must be a dictionary
  /// @param lkey             Name of the key which will contains the actual label index from the test dataset header
  /// @param test_header      Test dataset header
  /// @param pkey             Name of the key mapping to the probabilities
  /// @param probabilities    Actual probabilities to record. A column represents the result of one test exemplar.
  ///                         In a column, each row represents the probabilities for the train class matching the index
  /// @param wkey             Name of the key mapping to the weights
  /// @param weighs           A row vector indicating the "weights" associated to the probability.
  inline void record_results(Json::Value& out,
                      std::string const& lkey, DatasetHeader const& test_header,
                      std::string const& pkey, arma::mat const& probabilities,
                      std::string const& wkey, arma::rowvec const& weights
                      ) {
    // Record JSON
    const size_t test_top = test_header.size();
    Json::Value result_probas;
    Json::Value result_weights;
    Json::Value result_truelabels;
    // For each query (test instance)
    for (size_t query = 0; query<test_top; ++query) {
      // Store the probabilities
      result_probas.append(utils::to_json(probabilities.col(query)));
      // Store the weight associated with the probabilities
      result_weights.append(Json::Value(weights[query]));
      // Store the true label
      std::string true_l = test_header.labels()[query].value();
      size_t true_label_idx = test_header.label_to_index().at(true_l);
      result_truelabels.append(true_label_idx);
    }
    out[lkey] = result_truelabels;
    out[pkey] = result_probas;
    out[wkey] = result_weights;
  }

} // end of namespace tempo::classifier