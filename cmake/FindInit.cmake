# Adapted from https://gitlab.inria.fr/solverstack/morse_cmake/blob/0cd3b224949c838f0073d1cf545ab9ee6afd00a0/modules/find/FindInit.cmake

###
#
# @copyright (c) 2019 Inria. All rights reserved.
#
###
#
#  @file FindInit.cmake
#
#  @project MORSE
#  MORSE is a software package provided by:
#     Inria Bordeaux - Sud-Ouest,
#     Univ. of Tennessee,
#     King Abdullah Univesity of Science and Technology
#     Univ. of California Berkeley,
#     Univ. of Colorado Denver.
#
#  @version 1.0.0
#  @author Florent Pruvost
#  @date 24-04-2018
#
###


# This include is required to check symbols of libs
include(CheckFunctionExists)

# This include is required to check defines in headers
include(CheckIncludeFiles)

# clean these variables before using them in CMAKE_REQUIRED_* variables in
# check_function_exists
macro(finds_remove_duplicates)
  if (REQUIRED_DEFINITIONS)
    list(REMOVE_DUPLICATES REQUIRED_DEFINITIONS)
  endif()
  if (REQUIRED_INCDIRS)
    list(REMOVE_DUPLICATES REQUIRED_INCDIRS)
  endif()
  if (REQUIRED_FLAGS)
    list(REMOVE_DUPLICATES REQUIRED_FLAGS)
  endif()
  if (REQUIRED_LDFLAGS)
    list(REMOVE_DUPLICATES REQUIRED_LDFLAGS)
  endif()
  if (REQUIRED_LIBS)
    list(REVERSE REQUIRED_LIBS)
    list(REMOVE_DUPLICATES REQUIRED_LIBS)
    list(REVERSE REQUIRED_LIBS)
  endif()
endmacro()


# To find headers and libs
# Some macros to print status when search for headers and libs
include(PrintFindStatus)

function(FindHeader _libname _header_to_find)

  # save _libname upper and lower case
  string(TOUPPER ${_libname} LIBNAME)
  string(TOLOWER ${_libname} libname)

  # Looking for include
  # -------------------

  # Add system include paths to search include
  # ------------------------------------------
  unset(_inc_env)
  if(WIN32)
    string(REPLACE ":" ";" _inc_env "$ENV{INCLUDE}")
  else()
    string(REPLACE ":" ";" _path_env "$ENV{INCLUDE}")
    list(APPEND _inc_env "${_path_env}")
    string(REPLACE ":" ";" _path_env "$ENV{C_INCLUDE_PATH}")
    list(APPEND _inc_env "${_path_env}")
    string(REPLACE ":" ";" _path_env "$ENV{CPATH}")
    list(APPEND _inc_env "${_path_env}")
    string(REPLACE ":" ";" _path_env "$ENV{INCLUDE_PATH}")
    list(APPEND _inc_env "${_path_env}")
  endif()
  list(APPEND _inc_env "${CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES}")
  list(REMOVE_DUPLICATES _inc_env)


  # Try to find the _header_to_find in the given paths
  # --------------------------------------------------
  # call cmake macro to find the header path
  if(${LIBNAME}_INCDIR)
    set(${LIBNAME}_${_header_to_find}_DIRS "${LIBNAME}_${_header_to_find}_DIRS-NOTFOUND")
    find_path(${LIBNAME}_${_header_to_find}_DIRS
      NAMES ${_header_to_find}
      HINTS ${${LIBNAME}_INCDIR})
  elseif(${LIBNAME}_DIR)
    set(${LIBNAME}_${_header_to_find}_DIRS "${LIBNAME}_${_header_to_find}_DIRS-NOTFOUND")
    find_path(${LIBNAME}_${_header_to_find}_DIRS
      NAMES ${_header_to_find}
      HINTS ${${LIBNAME}_DIR}
      PATH_SUFFIXES include)
  else()
    set(${LIBNAME}_${_header_to_find}_DIRS "${LIBNAME}_${_header_to_find}_DIRS-NOTFOUND")
    find_path(${LIBNAME}_${_header_to_find}_DIRS
      NAMES ${_header_to_find}
      HINTS ${_inc_env})
  endif()
  mark_as_advanced(${LIBNAME}_${_header_to_find}_DIRS)

  # Print status if not found
  # -------------------------
  if (NOT ${LIBNAME}_${_header_to_find}_DIRS)
    Print_Find_Header_Status(${libname} ${_header_to_find})
  endif ()

endfunction(FindHeader)


# Transform relative path into absolute path for libraries
# lib_list (input/output): the name of the CMake variable containing libraries, e.g. BLAS_LIBRARIES
# hints_paths (input): additional paths to add when looking for libraries
macro(LIBRARIES_ABSOLUTE_PATH lib_list hints_paths)
  # collect environment paths to dig 
  list(APPEND _lib_env "$ENV{LIBRARY_PATH}")
  if(WIN32)
    string(REPLACE ":" ";" _lib_env2 "$ENV{LIB}")
  elseif(APPLE)
    string(REPLACE ":" ";" _lib_env2 "$ENV{DYLD_LIBRARY_PATH}")
  else()
    string(REPLACE ":" ";" _lib_env2 "$ENV{LD_LIBRARY_PATH}")
  endif()
  list(APPEND _lib_env "${_lib_env2}")
  list(APPEND _lib_env "${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}")
  # copy the lib list 
  set (${lib_list}_COPY "${${lib_list}}")
  # reset the lib list to populate
  set(${lib_list} "")
  foreach(_library ${${lib_list}_COPY})
    if(EXISTS "${_library}")
      # if already an absolute path, nothing special to do
      list(APPEND ${lib_list} ${_library})
    else()
      # replace pattern -lfoo -> foo
      string(REGEX REPLACE "^-l" "" _library "${_library}")
      # remove extensions if exist
      get_filename_component(_ext "${_library}" EXT)
      set(_lib_extensions ".so" ".a" ".dyld" ".dll")
      list(FIND _lib_extensions "${_ext}" _index)
      if (${_index} GREATER -1)
        get_filename_component(_library "${_library}" NAME_WE)
      endif()
      # try to find the lib
      find_library(_library_path NAMES ${_library} HINTS ${hints_paths} ${_lib_env})
      if (_library_path)
          list(APPEND ${lib_list} ${_library_path})
      else()
          message(FATAL_ERROR "Dependency of ${lib_list} '${_library}' NOT FOUND")
      endif()
      unset(_library_path CACHE)
    endif()
  endforeach()
endmacro()

# Some macros to print status when search for headers and libs
include(PrintFindStatus)

##
## @end file FindInit.cmake
##
