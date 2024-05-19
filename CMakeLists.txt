cmake_minimum_required(VERSION 3.20.0)

IF(WIN32)
    set(VCPKG_TARGET_TRIPLET x64-windows-static)
ENDIF()

IF(APPLE)
    SET(CMAKE_C_COMPILER "clang")
    SET(CMAKE_CXX_COMPILER "clang++")
ENDIF()

find_package(Git REQUIRED)
message("Git executable: ${GIT_EXECUTABLE}")

IF(EXISTS ".git")
    SET(GIT_COMMAND_EXECUTED "{GIT_EXECUTABLE} submodule update --init --recursive")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} RESULT_VARIABLE GIT_SUBMOD_RESULT)
ELSE()
    SET(GIT_COMMIT_HASH "a1212c93c")
    SET(GIT_COMMAND_EXECUTED "${GIT_EXECUTABLE} clone https://github.com/Microsoft/vcpkg.git")
    execute_process(COMMAND ${GIT_EXECUTABLE} clone https://github.com/Microsoft/vcpkg.git
                    WORKING_DIRECTORY "." RESULT_VARIABLE GIT_SUBMOD_RESULT)

    IF(NOT GIT_SUBMOD_RESULT EQUAL "0")
        message(FATAL_ERROR "${GIT_COMMAND_EXECUTED} failed with ${GIT_SUBMOD_RESULT}.")
    ENDIF()

    SET(GIT_COMMAND_EXECUTED "${GIT_EXECUTABLE} checkout ${GIT_COMMIT_HASH}")
    execute_process(COMMAND ${GIT_EXECUTABLE} checkout ${GIT_COMMIT_HASH}
                    WORKING_DIRECTORY "vcpkg" RESULT_VARIABLE GIT_SUBMOD_RESULT)

ENDIF()

IF(NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR "${GIT_COMMAND_EXECUTED} failed with ${GIT_SUBMOD_RESULT}, please checkout submodules.")
ENDIF()

set(CMAKE_TOOLCHAIN_FILE "vcpkg/scripts/buildsystems/vcpkg.cmake")

project(pybnesian VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)
ADD_DEFINITIONS("-DVERSION_INFO=${SKBUILD_PROJECT_VERSION}")


set(CMAKE_CXX_STANDARD 17)

IF(MSVC)
    SET(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    ADD_DEFINITIONS("-DNOGDI")
ENDIF()

set(PYBIND11_NEWPYTHON ON)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

message("Major version: ${Python_VERSION_MAJOR}")
message("Minor version: ${Python_VERSION_MINOR}")

add_definitions(-DPYTHON_VERSION_MAJOR=${Python_VERSION_MAJOR} -DPYTHON_VERSION_MINOR=${Python_VERSION_MINOR})

IF(WIN32)
    SET(SCRIPT_PREFIX "")
    SET(SCRIPT_EXTENSION "bat")
ELSEIF(UNIX)
    SET(SCRIPT_PREFIX "./")
    SET(SCRIPT_EXTENSION "sh")
ENDIF()

execute_process(COMMAND python expand_sources.py RESULT_VARIABLE EXPAND_SOURCES_RESULT)

IF(NOT EXPAND_SOURCES_RESULT EQUAL "0")
    message(FATAL_ERROR "$python expand_sources.py failed with ${EXPAND_SOURCES_RESULT}")
ENDIF()

execute_process(COMMAND ${SCRIPT_PREFIX}bootstrap-vcpkg.${SCRIPT_EXTENSION} WORKING_DIRECTORY "vcpkg" RESULT_VARIABLE VCPKG_BOOTSTRAP_RESULT)

IF(NOT VCPKG_BOOTSTRAP_RESULT EQUAL "0")
    message(FATAL_ERROR "${SCRIPT_PREFIX}bootstrap-vcpkg.${SCRIPT_EXTENSION} failed with ${VCPKG_BOOTSTRAP_RESULT}")
ENDIF()

execute_process(COMMAND ${SCRIPT_PREFIX}vcpkg install WORKING_DIRECTORY "vcpkg" RESULT_VARIABLE VCPKG_INSTALL_RESULT)

IF(NOT VCPKG_INSTALL_RESULT EQUAL "0")
    message(FATAL_ERROR "${SCRIPT_PREFIX}vcpkg install failed with ${VCPKG_INSTALL_RESULT}")
ENDIF()

find_package(Arrow CONFIG REQUIRED)
message("Arrow found: ${Arrow_FOUND}")
message("Arrow include: ${ARROW_INCLUDE_DIR}")

find_package(NLopt CONFIG REQUIRED)
message("NLOPT found: ${NLOPT_FOUND}")
message("NLOPT include: ${NLOPT_INCLUDE_DIR}")
message("NLOPT includes: ${NLOPT_INCLUDE_DIRS}")

find_package(libfort CONFIG REQUIRED)
message("LIBFORT found: ${LIBFORT_FOUND}")
message("libfort include: ${LIBFORT_INCLUDE_DIR}")
message("libfort includes: ${LIBFORT_INCLUDE_DIRS}")

find_package(Boost REQUIRED COMPONENTS math dynamic_bitset)

find_package(OpenCL REQUIRED)

pybind11_add_module(__init__ "pybnesian/lib.cpp"
                              "pybnesian/pybindings/pybindings_dataset.cpp"
                              "pybnesian/pybindings/pybindings_kde.cpp"
                              "pybnesian/pybindings/pybindings_factors.cpp"
                              "pybnesian/pybindings/pybindings_graph.cpp"
                              "pybnesian/pybindings/pybindings_models.cpp"
                              "pybnesian/pybindings/pybindings_learning/pybindings_learning.cpp"
                              "pybnesian/pybindings/pybindings_learning/pybindings_scores.cpp"
                              "pybnesian/pybindings/pybindings_learning/pybindings_independences.cpp"
                              "pybnesian/pybindings/pybindings_learning/pybindings_parameters.cpp"
                              "pybnesian/pybindings/pybindings_learning/pybindings_mle.cpp"
                              "pybnesian/pybindings/pybindings_learning/pybindings_operators.cpp"
                              "pybnesian/pybindings/pybindings_learning/pybindings_algorithms.cpp"
                              "pybnesian/kde/KDE.cpp"
                              "pybnesian/kde/ProductKDE.cpp"
                              "pybnesian/kde/UCV.cpp"
                              "pybnesian/factors/continuous/LinearGaussianCPD.cpp"
                              "pybnesian/factors/continuous/CKDE.cpp"
                              "pybnesian/factors/discrete/DiscreteFactor.cpp"
                              "pybnesian/factors/discrete/discrete_indices.cpp"
                              "pybnesian/dataset/dataset.cpp"
                              "pybnesian/dataset/dynamic_dataset.cpp"
                              "pybnesian/dataset/crossvalidation_adaptator.cpp"
                              "pybnesian/dataset/holdout_adaptator.cpp"
                              "pybnesian/util/arrow_types.cpp"
                              "pybnesian/util/bit_util.cpp"
                              "pybnesian/util/validate_options.cpp"
                              "pybnesian/util/validate_whitelists.cpp"
                              "pybnesian/util/temporal.cpp"
                              "pybnesian/util/rpoly.cpp"
                              "pybnesian/util/vech_ops.cpp"
                              "pybnesian/util/pickle.cpp"
                              "pybnesian/util/util_types.cpp"
                              "pybnesian/kdtree/kdtree.cpp"
                              "pybnesian/learning/operators/operators.cpp"
                              "pybnesian/learning/algorithms/hillclimbing.cpp"
                              "pybnesian/learning/algorithms/pc.cpp"
                              "pybnesian/learning/algorithms/mmpc.cpp"
                              "pybnesian/learning/algorithms/mmhc.cpp"
                              "pybnesian/learning/algorithms/dmmhc.cpp"
                              "pybnesian/learning/independences/continuous/linearcorrelation.cpp"
                              "pybnesian/learning/independences/continuous/mutual_information.cpp"
                              "pybnesian/learning/independences/continuous/RCoT.cpp"
                              "pybnesian/learning/independences/discrete/chi_square.cpp"
                              "pybnesian/learning/independences/hybrid/mutual_information.cpp"
                              "pybnesian/learning/parameters/mle_LinearGaussianCPD.cpp"
                              "pybnesian/learning/parameters/mle_DiscreteFactor.cpp"
                              "pybnesian/learning/scores/bic.cpp"
                              "pybnesian/learning/scores/bge.cpp"
                              "pybnesian/learning/scores/bde.cpp"
                              "pybnesian/learning/scores/cv_likelihood.cpp"
                              "pybnesian/learning/scores/holdout_likelihood.cpp"
                              "pybnesian/graph/generic_graph.cpp"
                              "pybnesian/models/BayesianNetwork.cpp"
                              "pybnesian/models/GaussianNetwork.cpp"
                              "pybnesian/models/SemiparametricBN.cpp"
                              "pybnesian/models/KDENetwork.cpp"
                              "pybnesian/models/DiscreteBN.cpp"
                              "pybnesian/models/HomogeneousBN.cpp"
                              "pybnesian/models/HeterogeneousBN.cpp"
                              "pybnesian/models/CLGNetwork.cpp"
                              "pybnesian/models/DynamicBayesianNetwork.cpp"
                              "pybnesian/opencl/opencl_config.cpp")


target_include_directories(__init__ PRIVATE "pybnesian")
target_include_directories(__init__ SYSTEM PRIVATE "lib/eigen-3.3.7" "lib/indicators" "lib/OpenCL")

target_link_libraries(__init__ PRIVATE Arrow::arrow_static OpenCL::OpenCL NLopt::nlopt libfort::fort Boost::dynamic_bitset Boost::math)

install(TARGETS __init__ LIBRARY DESTINATION ./pybnesian)

