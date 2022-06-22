add_executable(scratch)
target_sources(scratch PRIVATE scratch.cpp pch.h)
target_link_libraries(scratch PRIVATE libtempo)
target_precompile_headers(scratch PRIVATE pch.h)
