cmake_minimum_required(VERSION 3.13.0)

if(NOT CMAKE_BUILD_TYPE) 
    set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

project(pspaQR)

option(SPAQR_USE_MKL "Build with MKL" ON)
option(SPAQR_USE_METIS "Build with METIS" ON)
option(SPAQR_USE_HSL "Build with HSL" OFF)

### Find dependencies
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

### BLAS
if(SPAQR_USE_MKL)
    set(BLAS_DIR $ENV{MKLROOT})
    set(BLA_VENDOR "Intel10_64lp")
    message(STATUS "Using MKL")
else()
    set(BLA_VENDOR Open)
    message(STATUS "Using OpenBlas")
endif()


find_package(BLAS REQUIRED)
find_package(LAPACKE REQUIRED)

### EIGEN
set(EIGEN3_INCLUDE_DIR ${SOFTROOT}/eigen)
find_package(Eigen3 3.3 REQUIRED)
add_definitions(-DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE) 
if(SPAQR_USE_MKL)
    add_definitions(-DUSE_MKL -DEIGEN_USE_MKL_ALL)
endif() 

### Setup compile flags
enable_language(CXX)
set (CMAKE_CXX_STANDARD 14)

execute_process(COMMAND ${CMAKE_C_COMPILER} -fuse-ld=lld -Wl,--version OUTPUT_VARIABLE lldOut ERROR_QUIET)
execute_process(COMMAND ${CMAKE_C_COMPILER} -fuse-ld=gold -Wl,--version OUTPUT_VARIABLE goldOut ERROR_QUIET)
if("${lldOut}" MATCHES "LLD")
    add_compile_options(-fuse-ld=lld)
    message(STATUS "Using LLD linker")
elseif("${goldOut}" MATCHES "GNU gold")
    add_compile_options(-fuse-ld=gold)
    message(STATUS "Using GNU gold linker")
else()
    message(STATUS "Using the default system linker")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O0 -Wall -Wextra -Wnull-dereference -Wno-sign-compare -Wno-int-in-bool-context -fsanitize=address)
else()
    add_compile_options(-O3 -Wall -Wno-unused-variable -mtune=native -march=native -fomit-frame-pointer -funroll-loops -ffast-math)
    add_definitions(-DEIGEN_NO_DEBUG -DNDEBUG)
endif()

### Add TaskTorrent
add_subdirectory(${TASKTORRENT_ROOT_DIR} ${CMAKE_BINARY_DIR}/ttor)

# METIS AND HSL
if(SPAQR_USE_METIS)
    add_definitions(-DUSE_METIS)
	set(ENV{PART_INCLUDE_DIRS} ${METIS_INC_DIR})
	set(ENV{PART_LIBRARIES} ${METIS_LIB})
	message(STATUS "Using Metis")
else()
	set(ENV{PART_INCLUDE_DIRS} ${SOFTROOT}/patoh/Darwin-x86_64)
	set(ENV{PART_LIBRARIES} ${SOFTROOT}/patoh/Darwin-x86_64/libpatoh.a)
	message(STATUS "Using PaToH")
endif()

if(SPAQR_USE_HSL)
    add_definitions(-DHSL_AVAIL)
	set(ENV{HSL_INCLUDE_DIRS} ${SOFTROOT}/hsl_mc64-2.3.1/include)
	set(ENV{HSL_LIBRARIES} ${SOFTROOT}/hsl_mc64-2.3.1/src/libhsl_mc64.a)
    set(ENV{FORTRAN_INCLUDE_DIRS} /usr/local/gfortran/include)
    set(ENV{FORTRAN_LIBRARIES} /usr/local/gfortran/lib/libgfotran.a)
	message(STATUS "Using HSL")
endif()

if(TTOR_USE_HWLOC)
    add_definitions(-DUSE_HWLOC)
    set(ENV{HWLOC_INCLUDE_DIRS} ${SOFTROOT}/hwloc/build/install/include)
    set(ENV{HWLOC_LIBRARIES} ${SOFTROOT}/hwloc/build/install/lib/libhwloc.a)
    message(STATUS "Using HWLOC")
endif()

add_compile_options(${BLAS_COMPILER_FLAGS})
add_link_options(${BLAS_LINKER_FLAGS})

### Add build targets
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_subdirectory(src)

### Build executable
add_executable(spaQR $<TARGET_OBJECTS:spaQR_lib> pspaQR.cpp)
target_include_directories(spaQR
    PUBLIC
        ../include
        ../external
)

target_include_directories(spaQR
    SYSTEM PRIVATE ${BLAS_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        $ENV{PART_INCLUDE_DIRS}
        $ENV{HSL_INCLUDE_DIRS}
        $ENV{FORTRAN_INCLUDE_DIRS}
)

target_link_libraries(spaQR
    PRIVATE ${BLAS_LIBRARIES}
        $ENV{PART_LIBRARIES}
        $ENV{HSL_LIBRARIES}
        $ENV{FORTRAN_LIBRARIES}
        TaskTorrent
        spaQR_lib
)

add_subdirectory(tests)

### Build speedups
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "Using ccache")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
else()
    message(STATUS "Skipping ccache")
endif()


