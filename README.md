# ProximityForest 2.0
**Proximity Forest 2.0: A new effective and scalable similarity-based classifier for time series**

*Preprint*: []() 

> <div align="justify">Time series classification (TSC) is a challenging task due to the diversity of types of feature that may be relevant for different classification tasks, including trends, variance, frequency, magnitude, and various patterns. To address this challenge, several alternative classes of approach have been developed, including similarity-based, features and intervals, shapelets, dictionary, kernel, neural network, and hybrid approaches. While kernel, neural network, and hybrid approaches perform well overall, some specialized approaches are better suited for specific tasks. In this paper, we propose a new similarity-based classifier, Proximity Forest version 2.0 (PF 2.0), which outperforms  previous state-of-the-art similarity-based classifiers across the UCR benchmark and outperforms state-of-the-art kernel, neural network, and hybrid methods on specific datasets in the benchmark that are best addressed by similarity-base methods.  PF 2.0 incorporates three recent advances in time series similarity measures --- (1) computationally efficient early abandoning and pruning to speedup elastic similarity computations; (2) a new elastic similarity measure, Amerced Dynamic Time Warping (ADTW); and (3) cost function tuning. It rationalizes the set of similarity measures employed, reducing the eight base measures of the original PF to three and using the first derivative transform with all similarity measures, rather than a limited subset. We have implemented both PF 1.0 and PF 2.0 in a single C++ framework, making the PF framework more efficient.</div>

## Reference
If you use any part of this work, please cite:
```
@article{Herrmann2023PF2,
  title={{Proximity Forest 2.0}: A new effective and scalable similarity-based classifier for time series},
  author={Herrmann, Matthieu and Tan, Chang Wei and Salehi, Mahsa and Webb, Geoffrey I},
  year={2023},
  journal={arxiv:2023}
}
```

## Code
```
<EXEC_PATH> <path_to_csv_dataset> -t <nb_trees> -c <nb_candidates> -p <nb_threads> --pfc <pf_configs> -o <output_path>"

pf_configs: 
- pf2
- any combination of distances such as DA:DTWFull:DTW:WDTW:LCSS:MSM:ERP:TWE:ADTW in this format
``` 
