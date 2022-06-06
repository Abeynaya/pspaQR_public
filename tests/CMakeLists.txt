message(STATUS "test_pspaQR")
### Build executable
add_executable(test_pspaQR $<TARGET_OBJECTS:spaQR_lib> test_pspaQR.cpp)
target_include_directories(test_pspaQR
    PUBLIC
        ../include
        ../external
)

target_include_directories(test_pspaQR
    SYSTEM PRIVATE ${BLAS_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        $ENV{PART_INCLUDE_DIRS}
        $ENV{HSL_INCLUDE_DIRS}
        $ENV{FORTRAN_INCLUDE_DIRS}
)

target_link_libraries(test_pspaQR
    PRIVATE ${BLAS_LIBRARIES}
        $ENV{PART_LIBRARIES}
        $ENV{HSL_LIBRARIES}
        $ENV{FORTRAN_LIBRARIES}
        TaskTorrent
        spaQR_lib
)