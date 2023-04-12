add_library(tclap INTERFACE)
target_include_directories(tclap INTERFACE external/tclap-1.2.5/include)

add_subdirectory(PF2)

# add_subdirectory(scratch)
# add_subdirectory(UCRInfo)
