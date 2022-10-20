add_executable(loocv)
target_sources(loocv
        PRIVATE
        main.cpp
        iloocv_adtw.cpp
        iloocv_dtw.cpp
        iloocv_wdtw.cpp
        )

target_link_libraries(loocv PRIVATE libtempo)
