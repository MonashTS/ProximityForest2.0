add_executable(pf)
target_sources(pf PRIVATE main.cpp pch.h)
target_link_libraries(pf PUBLIC libtempo)
target_precompile_headers(pf PRIVATE pch.h)