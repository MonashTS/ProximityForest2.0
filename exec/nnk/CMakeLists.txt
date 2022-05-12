add_executable(nnk)
target_sources(nnk PRIVATE main.cpp pch.h)
target_link_libraries(nnk PRIVATE libtempo)
target_precompile_headers(pf PRIVATE pch.h)
