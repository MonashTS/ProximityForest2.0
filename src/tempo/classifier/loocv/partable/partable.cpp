#include "partable.hpp"

#include <tempo/utils/utils.hpp>

namespace tempo::classifier::nn1loocv {

  std::tuple<std::vector<size_t>, size_t> partable(
    dist_ft distance,
    distUB_ft distanceUB,
    DatasetHeader const& train_header,
    size_t nbparams,
    size_t nbthreads
  ) {
    const size_t NBLINE = train_header.size();
    const size_t NBCOL = nbparams;
    const size_t NBCELL = NBLINE*NBCOL;

    // NNTable: one line per series, |params| column.
    // At each column, register the closest NN ID and the associated distance
    std::vector<NNC> NNTable(NBCELL);
    for (size_t i = 0; i<NBCELL; ++i) {
      NNTable[i].NNindex = NBLINE;
      NNTable[i].NNdistance = tempo::utils::PINF;
    }

    // Ordered set of added series, sorted by their (approximated) descending NN distance in NNTable[T, last]
    // i.e. the first series has a large distance to its NN, and the last one has a small distance to its NN.
    std::vector<size_t> intable;
    intable.reserve(NBLINE);

    auto update = [&](size_t table_idx, size_t nnindex, double nndist) {
      auto& nn = NNTable[table_idx];
      std::lock_guard lock(nn.mutex);
      if (nndist<nn.NNdistance) {
        nn.NNindex = nnindex;
        nn.NNdistance = nndist;
      }
    };

    // --- --- --- Add the first pair
    {
      // Start with the upper bound
      {
        const size_t LASTP = nbparams - 1;
        FloatType UB = distanceUB(0, 1, LASTP);
        FloatType d_last = distance(0, 1, LASTP, UB);
        update(0*NBCOL + NBCOL - 1, 1, d_last);
        update(1*NBCOL + NBCOL - 1, 0, d_last);
      }
      // Complete the table cutting with the UB (we are not going to EA anything here)
      for (int pidx = (int)NBCOL - 2; pidx>=0; --pidx) {
        FloatType UB = NNTable[0*NBCOL + pidx + 1].NNdistance;
        FloatType d = distance(0, 1, pidx, UB);
        update(0*NBCOL + pidx, 1, d);
        update(1*NBCOL + pidx, 0, d);
      }
      intable.push_back(0);
      intable.push_back(1);
    }

    tempo::utils::ParTasks ptask;

    // --- --- --- Add the other series
    for (size_t S = 2; S<NBLINE; ++S) {

      // --- Complete with other series already in the table
      // This loop generates a set of task (one per Ti)
      for (size_t Ti : intable) {
        // --- Define the tasks
        auto task = [&, Ti]() {
          // Max bound: if above this, S and T cannot be each other NN
          const FloatType
            dmax = std::max(NNTable[S*NBCOL + NBCOL - 1].NNdistance, NNTable[Ti*NBCOL + NBCOL - 1].NNdistance);
          // Start the process with the first parameters, and no lower bound
          size_t Pi = 0;
          FloatType LB = 0;
          do {
            const FloatType d_S = NNTable[S*NBCOL + Pi].NNdistance;
            const FloatType d_Ti = NNTable[Ti*NBCOL + Pi].NNdistance;
            const FloatType d_nn = std::max(d_S, d_Ti);
            if (LB<d_nn) {
              const FloatType cutoff = std::min(dmax, distanceUB(S, Ti, Pi));
              const FloatType di = distance(S, Ti, Pi, cutoff);
              update(S*NBCOL + Pi, Ti, di);
              update(Ti*NBCOL + Pi, S, di);
              if (di==tempo::utils::PINF) { Pi = NBCOL; }
              else { LB = di; }
            }
            Pi++;
          } while (Pi<NBCOL);
        };
        // --- Add the tasks
        ptask.push_task(std::move(task));
      }

      // --- Execute the tasks in parallel
      ptask.execute(nbthreads);

      // --- Put S in intable, maintaining approximate descending order on NNTable[S*NBCOL+NBCOL-1]
      intable.push_back(S);
      for (size_t Tidx = intable.size() - 1; Tidx>=1; --Tidx) {
        const size_t Ti = intable[Tidx];
        const size_t Tiprev = intable[Tidx - 1];
        if (NNTable[Ti*NBCOL + NBCOL - 1].NNdistance>NNTable[Tiprev*NBCOL + NBCOL - 1].NNdistance) {
          std::swap(intable[Ti], intable[Tiprev]);
        }
      }

    }// End of Table filling


    // --- --- --- NNTable is full: find the param with the fewest error
    std::vector<size_t> result;
    size_t bestError = std::numeric_limits<size_t>::max();
    {
      for (size_t pidx = 0; pidx<NBCOL; ++pidx) {
        size_t nError = 0;
        for (size_t Ti = 0; Ti<NBLINE; Ti++) {
          if (train_header.label(Ti).value()!=train_header.label(NNTable[Ti*NBCOL + pidx].NNindex).value()) {
            nError++;
          }
        }
        //
        if (nError<bestError) {
          result.clear();
          result.push_back(pidx);
          bestError = nError;
        } else if (nError==bestError) { result.push_back(pidx); }
      }
    }

    return {result, NBLINE - bestError};
  }

}
