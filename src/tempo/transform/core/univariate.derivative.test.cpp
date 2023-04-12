#include <catch2/catch_test_macros.hpp>

#include "univariate.derivative.hpp"

#include <mock/mockseries.hpp>

#include <vector>

using F = double;
constexpr size_t nbitems = 500;
using namespace tempo::transform::core::univariate;

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Reference
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
namespace ref {

  void derive(F const *data, size_t length, F *output, size_t degree) {
    if (degree==0) {
      std::copy(data, data + length, output);
    } else {
      // D1
      ::derive<F, F const*, F*>(data, length, output);

      // D2 and after: need a temporary variable - derivative not computed "in place"
      if (degree>1) {
        std::unique_ptr<F[]> tmp(new F[length]);
        F *a = output;
        F *b = tmp.get();

        for (size_t d = 2; d<=degree; ++d) {
          ::derive<F, F const*, F*>(a, length, b);
          std::swap(a, b);
        }

        // Copy in the output evey two turns
        if (degree%2==0) {
          assert(b==output);
          std::copy(a, a + length, b);
        }
      }
    }
  }

} // End of namespace ref

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
TEST_CASE("Univariate Derivative", "[transform][univariate][derivative]") {
  Catch::StringMaker<F>::precision = 18;

  // Setup univariate with fixed length
  mock::Mocker mocker;
  const auto fset = mocker.vec_randvec(nbitems);

  const size_t length = mocker._fixl;

  //
  for (const auto& s : fset) {
    std::vector<double> d1(length);
    std::vector<double> d2(length);
    std::vector<double> d3(length);

    double *out = d1.data();
    derive<F, F const *, F *>(s.data(), length, out);

    out = d2.data();
    derive<F, F const *, F *>(d1.data(), length, out);

    out = d3.data();
    derive<F, F const *, F *>(d2.data(), length, out);

    std::vector<double> testmulti(length);
    out = testmulti.data();

    ref::derive(s.data(), length, out, 1);
    REQUIRE(std::memcmp(out, d1.data(), length)==0);

    ref::derive(s.data(), length, out, 2);
    REQUIRE(std::memcmp(out, d2.data(), length)==0);

    ref::derive(s.data(), length, out, 3);
    REQUIRE(std::memcmp(out, d3.data(), length)==0);

  }

}