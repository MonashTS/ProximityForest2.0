#pragma once

#include <functional>

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/dataset.hpp>
#include "tempo/distance/helpers.hpp"

namespace tempo::classifier::loocv::partable {

  /// Type of functions used to compute a distance between two time series while performing LOOCV.
  /// The function is indexed by a "parameter index" in the range [0, nb_params[ (see loocv_incparams below).
  /// It must internalise these parameters.
  /// @param pindex       Index parameter in the [0, nb_params[ range
  /// @param query        Index of the query in the train set
  /// @param candidate    Index of the candidate in the train set
  /// @param ub           Upper bound on the process (such as the distance of the 'best so far')
  using distance_fun_t = std::function<tempo::F(size_t pindex, size_t query_idx, size_t candidate_idx, tempo::F ub)>;

  /// Type of functions used to compute an upper bound of a distance between two time series.
  /// The function is indexed by a "parameter index" in the range [0, nb_params[ (see loocv_incparams below).
  /// It must internalise these parameters.
  /// @param pindex       Index parameter in the [0, nb_params[ range (see loocv_incparams below)
  /// @param query        Index of the query in the train set
  /// @param candidate    Index of the candidate in the train set
  using upperbound_fun_t = std::function<FloatType(size_t pindex, size_t query_idx, size_t candidate_idx)>;


  /** Search for the best parameter through LOOCV through parameters leading to increasing costs.
   * @param  nb_params      Number of parameters.
   *                        It is understood that the distance and UB functions find the current parameter value
   *                        by indexing in an array 'params' of parameter from 0 to nb_params-1, ordered such that
   *                            params[0] is a LB for params[1] ... for params[last]
   *                            i.e. for two time series A and B and a distance 'dist', we must have
   *                            dist(A, B, p[0]) <= dist(A, B, p[1]) ... <= dist(A, B, p[n])
   *                            In turns, params[last] is an upper bound for params[last-1] ... for p[0]
   *                        When searching over several parameters, such an ordering may not be possible.
   *                        However, it may be recovered after fixing one parameter.
   *                        In this case, simply call this function several time, searching over the unfixed parameters.
   * @param train_header    Header of the used train set
   * @param distance        How to compute a distance between two series, given a parameter index
   * @param upper_bound     How to compute an upper bound between two series, given a parameter index
   * @param nb_threads      The number of thread to use
   * @return (vector of best parameters' index, bestError)
   */
  std::tuple<std::vector<size_t>, size_t> loocv_incparams(
    size_t                        nb_params,
    tempo::DatasetHeader const&   train_header,
    distance_fun_t                distance,
    upperbound_fun_t              upper_bound,
    size_t                        nb_threads
  );

 // std::tuple<std::vector<size_t>, size_t>




} // End of namespace tempo::classifier::loocv_incparams::partable

