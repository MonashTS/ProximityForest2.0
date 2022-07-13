# Dev notes

## Matt TODO

### Distances to implement
* Kernel DTW https://arxiv.org/abs/1005.5141
* GRAIL http://people.cs.uchicago.edu/~jopa/Papers/PaparrizosVLDB2019.pdf
  * more a transform + ED kind of nn1



## IDE

* May 2022 - after upgrading to gcc12, clangd analysis used in the CLion IDE
fails on some '__builtin_operator_delete'.
  * See https://youtrack.jetbrains.com/issue/CPP-29091
  * The delete operator is behind an '#if __cpp_sized_deallocation',
    which isn't set by default in clangd
  * Fix: add '-fsized-deallocation' to clangd flags in Settings | Languages & Frameworks | C/C++ | Clangd