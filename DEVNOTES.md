# Developer notes

## Third Party

### Eigen

We use the [Eigen](https://eigen.tuxfamily.org) library to store time series.
Being a template library, Eigen sources can directly be used (with a -I flag).
However, we use cmake, for which Eigen provides a package.
In order to use this package, the sources must be installed.
We install them in the [third_party/Eigen](third_party/Eigen) folder.

When updating Eigen, be sure to rerun the installation, e.g.
```
mkdir   _build
cd      _build
cmake   .. -DCMAKE_INSTALL_PREFIX=/path/to/third_party/Eigen
```
And don't forget to update our cmake files!