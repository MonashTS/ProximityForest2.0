#pragma once

#include <tempo/predef.hpp>
#include <tempo/dataset/dataset.hpp>

namespace tempo::classifier {

  /// Classifier result for one test exemplar
  struct Result1 {

    /// A row vector representing the classification result of one test exemplar.
    /// In the row vector, the ith column represent the probability of the ith class,
    /// where the ith class is determined by a label encoder.
    arma::rowvec probabilities;

    /// Weight associated to the probabilities
    double weight{};

    Result1() = default;

    Result1(Result1 const&) = default;

    Result1& operator =(Result1 const&) = default;

    Result1(Result1&&) = default;

    Result1& operator =(Result1&&) = default;

    inline Result1(arma::rowvec&& proba, double weight) : probabilities(std::move(proba)), weight(weight) {}

    inline explicit Result1(size_t nbclasses) : probabilities(nbclasses, arma::fill::zeros), weight(0) {}

    /// Create a Result for 'cardinality' classes, with the index (coming from a label encoder) of 'proba_at_one'
    /// being 1.0, and all the other at 0. The weight is copied as is.
    static inline Result1 make_probabilities_one(size_t cardinality, EL proba_at_one, double weight) {
      arma::rowvec p(cardinality, arma::fill::zeros);
      p[proba_at_one] = 1.0;
      return Result1(std::move(p), weight);
    }

  };

  /// Classifier result for several test exemplars
  struct ResultN {

    /// A matrix representing the classification result of several test exemplar.
    /// A row represents the classification result of one test exemplar.
    /// In the row vector, the ith column represent the probability of the ith class,
    /// where the ith class is determined by a label encoder.
    arma::mat probabilities{};

    /// Column vector of length nb_test_exemplars, indicating the "weights" (can be used to indicate confidence)
    /// associated to each row from the probabilities matrix
    arma::colvec weight{};

    ResultN() = default;

    ResultN(ResultN const&) = default;

    ResultN& operator =(ResultN const&) = default;

    ResultN(ResultN&&) = default;

    ResultN& operator =(ResultN&&) = default;

    inline void append(Result1 const& res1) {
      size_t n_rows = probabilities.n_rows;
      probabilities.insert_rows(n_rows, res1.probabilities);
      weight.insert_rows(n_rows, res1.weight);
    }

    inline size_t nb_correct_01loss(DatasetHeader const& test_header, IndexSet const& test_iset, PRNG& prng) {
      size_t nb_correct = 0;

      for (size_t r{0}; r<probabilities.n_rows; ++r) {
        // Find max probabilities, break ties randomly
        auto row = probabilities.row(r);
        double maxv = row.max();
        std::vector<size_t> maxp;
        for (size_t c{0}; c<probabilities.n_cols; ++c) { if (row[c]==maxv) { maxp.push_back(c); }}
        // Predicted and true endocded label
        EL predicted_elabel = utils::pick_one(maxp, prng);
        EL true_elabel = test_header.label(test_iset[r]).value();
        // Compare
        if (predicted_elabel==true_elabel) { nb_correct++; }
      }

      return nb_correct;
    }

  };



  /*
  struct ResultN {
    /// Resulting (nb_test_exemplars x nb_classess) matrix of probabilities.
    /// Each row represents the result of one test exemplar, in the order found in the test set.
    /// Inside a row, each column represents the class probabilities.
    /// Column indexes match the classes according to a label encoder.
    arma::mat probabilities;

    /// Column vector of length nb_test_exemplars, indicating the "weights" (can be used to indicate confidence)
    /// associated to each row from the probabilities matrix
    arma::colvec weights;
  };
   */

  /* Old transposed version

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
   */

} // end of namespace tempo::classifier