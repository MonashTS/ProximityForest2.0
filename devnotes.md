# Dev notes

## IDE

* May 2022 - after upgrading to gcc12, clangd analysis used in the CLion IDE
fails on some '__builtin_operator_delete'.
  * See https://youtrack.jetbrains.com/issue/CPP-29091
  * The delete operator is behind an '#if __cpp_sized_deallocation',
    which isn't set by default in clangd
  * Fix: add '-fsized-deallocation' to clangd flags in Settings | Languages & Frameworks | C/C++ | Clangd