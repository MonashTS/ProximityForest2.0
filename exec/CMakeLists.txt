add_library(tclap INTERFACE)
target_include_directories(tclap INTERFACE external/tclap-1.2.5/include)

add_subdirectory(TSChief)
add_subdirectory(nn1)

# add_subdirectory(distance_loocv)
# add_subdirectory(nnkgridsearch)
# add_subdirectory(scratch)
# add_subdirectory(UCRInfo)
