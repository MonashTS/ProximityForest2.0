# Create the libtempo target
add_library(libtempo)
target_include_directories(libtempo PUBLIC ${CMAKE_CURRENT_LIST_DIR}/..)

# Use precompile header
target_precompile_headers(libtempo PRIVATE pch_std.hpp pch_lib.hpp predef.hpp)

# Gather the sources (including the header)
target_sources(libtempo
    PRIVATE
        pch_std.hpp
        pch_lib.hpp
        predef.hpp
)

# Recurse (also run for the tests)
#add_subdirectory(classifier)
add_subdirectory(distance)
add_subdirectory(reader)
add_subdirectory(transform)
add_subdirectory(utils)
add_subdirectory(dataset)


# Linking
target_link_libraries(libtempo PUBLIC ${MLPACK_LIBRARY} ${JSONCPP_LIBRARIES} ${ARMADILLO_LIBRARIES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(libtempo PUBLIC OpenMP::OpenMP_CXX)
endif()

