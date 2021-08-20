# Use an INTERFACE library to add option to target.
# To be used with target_link_library(target PRIVATE project_option)
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_20)
target_compile_options(project_options INTERFACE -Wall -Wextra)
target_compile_definitions(project_options INTERFACE $<$<CONFIG:DEBUG>:_GLIBCXX_ASSERTIONS>)
