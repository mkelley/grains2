# Build the bhmie library
cmake_minimum_required(VERSION 3.17.2...3.29)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C Fortran)

find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRED)

# F2PY headers
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c
          "import numpy.f2py; print(numpy.f2py.get_include())"
  OUTPUT_VARIABLE F2PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

add_library(fortranobject OBJECT "${F2PY_INCLUDE_DIR}/fortranobject.c")
target_link_libraries(fortranobject PUBLIC Python::NumPy)
target_include_directories(fortranobject PUBLIC "${F2PY_INCLUDE_DIR}")
set_property(TARGET fortranobject PROPERTY POSITION_INDEPENDENT_CODE ON)

add_custom_command(
  OUTPUT bhmiemodule.c bhmie-f2pywrappers.f
  DEPENDS bhmie/bhmie.f
  VERBATIM
  COMMAND "${Python_EXECUTABLE}" -m numpy.f2py
          "${CMAKE_CURRENT_SOURCE_DIR}/bhmie/bhmie.f" -m bhmie --lower)

python_add_library(
  bhmie MODULE "${CMAKE_CURRENT_BINARY_DIR}/bhmiemodule.c"
  "${CMAKE_CURRENT_BINARY_DIR}/bhmie-f2pywrappers.f"
  "${CMAKE_CURRENT_SOURCE_DIR}/bhmie/bhmie.f" WITH_SOABI)
target_link_libraries(bhmie PRIVATE fortranobject)

install(TARGETS bhmie DESTINATION "grains2")

add_custom_command(
  OUTPUT davintmodule.c davint-f2pywrappers.f
  DEPENDS
    davint/davint.f
    davint/fdump.f
    davint/i1mach.f
    davint/j4save.f
    davint/xercnt.f
    davint/xerhlt.f
    davint/xermsg.f
    davint/xerprn.f
    davint/xersve.f
    davint/xgetua.f
  VERBATIM
  COMMAND "${Python_EXECUTABLE}" -m numpy.f2py
          "${CMAKE_CURRENT_SOURCE_DIR}/davint/davint.f"
          "${CMAKE_CURRENT_SOURCE_DIR}/davint/fdump.f"
          "${CMAKE_CURRENT_SOURCE_DIR}/davint/i1mach.f"
          "${CMAKE_CURRENT_SOURCE_DIR}/davint/j4save.f"
          "${CMAKE_CURRENT_SOURCE_DIR}/davint/xercnt.f"
          "${CMAKE_CURRENT_SOURCE_DIR}/davint/xerhlt.f"
          "${CMAKE_CURRENT_SOURCE_DIR}/davint/xermsg.f"
          "${CMAKE_CURRENT_SOURCE_DIR}/davint/xerprn.f"
          "${CMAKE_CURRENT_SOURCE_DIR}/davint/xersve.f"
          "${CMAKE_CURRENT_SOURCE_DIR}/davint/xgetua.f"
          -m davint --lower)

python_add_library(
  davint MODULE "${CMAKE_CURRENT_BINARY_DIR}/davintmodule.c"
  "${CMAKE_CURRENT_BINARY_DIR}/davint-f2pywrappers.f"
  "${CMAKE_CURRENT_SOURCE_DIR}/davint/davint.f"
  "${CMAKE_CURRENT_SOURCE_DIR}/davint/fdump.f"
  "${CMAKE_CURRENT_SOURCE_DIR}/davint/i1mach.f"
  "${CMAKE_CURRENT_SOURCE_DIR}/davint/j4save.f"
  "${CMAKE_CURRENT_SOURCE_DIR}/davint/xercnt.f"
  "${CMAKE_CURRENT_SOURCE_DIR}/davint/xerhlt.f"
  "${CMAKE_CURRENT_SOURCE_DIR}/davint/xermsg.f"
  "${CMAKE_CURRENT_SOURCE_DIR}/davint/xerprn.f"
  "${CMAKE_CURRENT_SOURCE_DIR}/davint/xersve.f"
  "${CMAKE_CURRENT_SOURCE_DIR}/davint/xgetua.f"
   WITH_SOABI)
target_link_libraries(davint PRIVATE fortranobject)

install(TARGETS davint DESTINATION "grains2")
