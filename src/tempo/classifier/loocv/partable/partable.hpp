#pragma once

#include <functional>
#include <tempo/dataset/dts.hpp>

namespace tempo::classifier::nn1loocv {

  /// Given a query, a candidate, and a parameter index, compute the distance between the query and the candidate.
  /// Can be early abandoned with 'bsf' (best so far)
  using dist_ft = std::function<F(size_t query_idx, size_t candidate_idx, size_t param_idx, F bsf)>;

  /// Same as above, but must produce an Upper Bound (UB) of the distance.
  /// For elastic distances using a cost matrix, this is usually done by taking the diagonal of the cost matrix,
  /// i.e. a form of direct alignment (eventually completed along the last line/column for disparate lengths)
  using distUB_ft = std::function<F(size_t query_idx, size_t candidate_idx, size_t param_idx)>;

  /// Nearest Neighbour Cell
  /// A cell of our table, with its own mutex
  struct NNC {
    std::mutex mutex{};   // Lock/unlock for multithreading
    size_t NNindex{};     // Index of the NN
    F NNdistance{};       // Distance to the NN
  };

  /// Search for the best parameter through LOOCV
  /// @param distance A distance computation function of type 'dist_fb'.
  ///     Must capture the series, the actual distance, and the parameters, so that it can be called with indexes.
  /// @param distanceUB Simular as above, of type 'distUB_fb', producing an upper bound.
  /// @param nbtrain Number of train exemplars: the distances will be call with distance(i, j, p) with
  ///     0<=i<nbtrain, 0<=j<nbtrain, i!=j, and p a parameter index
  /// @param nbparams Number of parameters
  ///     It is assumed that the distance functions capture a mapping of parameter N->Internal Parameter,
  ///     usually with a vector of parameter p, with 0 <= n:N < nbparams.
  ///     It is assumed that the parameter are ordered such that the distance computed with p[0] is a LB for p[1], etc:
  ///         dist(a, b, 0) <= dist(a, b, 1) ... <= dist(a, b, n-1)
  ///     In turns, p[last] produces an upper bound for p[last-1] ... etc ... for p[0]
  /// @param nbthreads parallelize the process on nbthreads -
  ///     Note that if nbthreads<2, this is not the best method as we spend time taking/realising mutexes!
  /// @return (vector of best parameters' index, bestError)
  std::tuple<std::vector<size_t>, size_t> partable(
    dist_ft distance,
    distUB_ft distanceUB,
    DatasetHeader const& train_header,
    size_t nbparams,
    size_t nbthreads
  );



}