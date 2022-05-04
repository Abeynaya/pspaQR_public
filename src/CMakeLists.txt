### Create spaQR object library
add_library(spaQR_lib OBJECT)


target_include_directories(spaQR_lib
    PUBLIC
        ../include
        ../external
)

target_sources(spaQR_lib
    PRIVATE
        util.cpp
        partition.cpp
        cluster.cpp
        edge.cpp
        is.cpp
        operations.cpp
        toperations.cpp
        profile.cpp
        tree.cpp
        ptree.cpp
)

target_include_directories(spaQR_lib
    SYSTEM PRIVATE ${BLAS_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        $ENV{PART_INCLUDE_DIRS}
        $ENV{HSL_INCLUDE_DIRS}
        $ENV{FORTRAN_INCLUDE_DIRS}
)

find_package(MPI REQUIRED)
target_link_libraries(spaQR_lib PUBLIC MPI::MPI_CXX)

# TO-DO: Change to PRIVATE
# See error w/ object library linking https://cmake.org/cmake/help/v3.15/release/3.14.html#other-changes
target_link_libraries(spaQR_lib
    PUBLIC ${BLAS_LIBRARIES}
        $ENV{PART_LIBRARIES}
        $ENV{HSL_LIBRARIES}
        $ENV{FORTRAN_LIBRARIES}
        TaskTorrent
)

# add_library(spaQR_shared SHARED)


# target_include_directories(spaQR_shared
#     PUBLIC
#         ../include
#         ../external
# )

# target_sources(spaQR_shared
#     PRIVATE
#         util.cpp
#         partition.cpp
#         cluster.cpp
#         edge.cpp
#         is.cpp
#         operations.cpp
#         toperations.cpp
#         profile.cpp
#         tree.cpp
#         ptree.cpp
# )

# target_include_directories(spaQR_shared
#     SYSTEM PRIVATE ${BLAS_INCLUDE_DIRS}
#         ${EIGEN3_INCLUDE_DIR}
#         $ENV{PART_INCLUDE_DIRS}
#         $ENV{HSL_INCLUDE_DIRS}
#         $ENV{FORTRAN_INCLUDE_DIRS}
# )

# # TO-DO: Change to PRIVATE
# # See error w/ object library linking https://cmake.org/cmake/help/v3.15/release/3.14.html#other-changes
# target_link_libraries(spaQR_shared
#     PUBLIC ${BLAS_LIBRARIES}
#         $ENV{PART_LIBRARIES}
#         $ENV{HSL_LIBRARIES}
#         TaskTorrent
#         $ENV{FORTRAN_LIBRARIES}
# )