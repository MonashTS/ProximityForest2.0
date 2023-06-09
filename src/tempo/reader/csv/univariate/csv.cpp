#include "csv.template.cpp"

namespace tempo::reader::univariate {

  // Implementation through template explicit instantiation

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Double implementation
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  using F = double;
  template Result<F> read_csv<F>(std::istream&, LabelEncoder const&, CSVReaderParam const&);

} // End of namespace tempo::reader
