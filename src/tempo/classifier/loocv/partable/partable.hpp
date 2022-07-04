#pragma once

#include <functional>

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/dataset.hpp>

namespace tempo::classifier::loocv::partable {

  /// Type of functions used to compute a distance while performing LOOCV (i.e. at train time)
  /// The function must capture all that it is needed to perform its job given the parameter,
  /// such as the train set, the parameter sequence, etc...
  /// @param query        Index of the query in the train set
  /// @param candidate    Index of the candidate in the train set
  /// @param pidx         Index of The parameter currently being assessed
  /// @param ub           Upper bound on the process (such as the distance of the 'best so far')
  using distance_fun_t = std::function<tempo::F(size_t query_idx, size_t candidate_idx, size_t pidx, tempo::F ub)>;

  /// Type of functions used to compute an initial upper bound between two series
  /// @param query        Index of the query in the train set
  /// @param candidate    Index of the candidate in the train set
  /// @param pidx         Index of The parameter currently being assessed
  using upperbound_fun_t = std::function<FloatType(size_t query_idx, size_t candidate_idx, size_t pidx)>;

  /** Search for the best parameter through LOOCV
   * @param  nb_params      Number of parameters.
   *                        It is understood that the distance and UB functions find the current parameter value
   *                        by indexing in an array 'params' of parameter from 0 to nb_params-1, ordered such that
   *                            params[0] is a LB for params[1] ... for params[last]
   *                            p[0] <= p[1] ... <= p[n]
   *                            In turns, params[last] is an upper bound for params[last-1] ... for p[0]
   *                        When searching over several parameters, such an ordering may not be possible.
   *                        However, it may be recovered after fixing one parameter.
   *                        In this case, simply call this function several time, searching over the unfixed parameters.
   * @param train_header    Header of the used train set
   * @param distance        How to compute a distance between two series, given a parameter index
   * @param diagUBdistance  How to compute an upper bound between two series, given a parameter index
   * @param nb_threads      The number of thread to use
   * @return (vector of best parameters' index, bestError)
   */
  std::tuple<std::vector<size_t>, size_t> loocv(
    size_t nb_params,
    tempo::DatasetHeader const& train_header,
    distance_fun_t distance,
    upperbound_fun_t diagUBdistance,
    size_t nb_threads
  );

} // End of namespace tempo::classifier::loocv::partable

