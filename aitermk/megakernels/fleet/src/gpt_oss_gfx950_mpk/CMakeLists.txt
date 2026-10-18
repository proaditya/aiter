cmake_minimum_required(VERSION 3.25)

# Fleet MPK GPT-OSS 120B gfx950 providers: the fleet persistent megakernel (provider-owned prefill, KV cache and
# weights), one DSO per build configuration under configs/<name>. Configure this directory on its own
# (cmake -S aitermk/megakernels/fleet/src/gpt_oss_gfx950_mpk), or add it from a parent project.
#
# Each DSO is compiled with exactly the compile command its configuration was validated with (configs/<name>/
# compile_flags.txt, recorded by the fleet exporter), plus -DMPK_QUIET_FWDPASS for serving (FLEET_MPK_QUIET_FWDPASS):
# hipcc from FLEET_MPK_ROCM_ROOT with that root's HIP headers and device libraries. CMake's HIP language is not used
# because it would substitute the build's own ROCm headers and compiler flags, and a different compiler gives
# different (not bit-exact) device code.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(aitermk_fleet_mpk_gpt_oss_gfx950 LANGUAGES CXX)
    include(GNUInstallDirs)
endif()

set(FLEET_MPK_ROCM_ROOT "" CACHE PATH
    "ROCm root whose bin/hipcc, include, llvm and amdgcn/bitcode compile the providers (validated: ROCm 7.0)")
set(FLEET_MPK_CONFIGS "ns;eagle3_b6;dflash_b8" CACHE STRING "fleet MPK provider configurations to build")
set(FLEET_MPK_CONFIG_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/configs" CACHE PATH
    "Directory holding one <config>/ per provider (kernel.hip, task_graph.json, compile_flags.txt, ...): this
    repository's configs, or a fresh export imported with tools/import_export.py")
option(FLEET_MPK_SAVE_TEMPS "Keep each DSO's device assembly (--save-temps) in its build directory" OFF)
option(FLEET_MPK_QUIET_FWDPASS
    "Compile with -DMPK_QUIET_FWDPASS: no per-pass [FWD_PASS] lines during or at the end of a launch (the
    [FWD_PASS_TOTAL] summary stays). Device printf goes through the host, and the next request waited for it" ON)
set(FLEET_CK_DIR "" CACHE PATH
    "Composable Kernel root at the Fleet pin, ROCm/composable_kernel d8ee107a47d8 (the fleet megakernel
    repository's deps/composable_kernel; aiter/3rdparty/composable_kernel_fleet in a fleet workspace)")

if(NOT FLEET_MPK_ROCM_ROOT)
    if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
        set(FLEET_MPK_ROCM_ROOT "$ENV{ROCM_PATH}")
    else()
        set(FLEET_MPK_ROCM_ROOT /opt/rocm)
    endif()
    message(WARNING "FLEET_MPK_ROCM_ROOT unset; using ${FLEET_MPK_ROCM_ROOT}. The configs were validated with "
                    "ROCm 7.0; another compiler builds different device code.")
endif()
if(NOT EXISTS "${FLEET_MPK_ROCM_ROOT}/bin/hipcc")
    message(FATAL_ERROR "FLEET_MPK_ROCM_ROOT=${FLEET_MPK_ROCM_ROOT} has no bin/hipcc")
endif()
if(NOT FLEET_CK_DIR AND DEFINED ENV{FLEET_CK_DIR} AND NOT "$ENV{FLEET_CK_DIR}" STREQUAL "")
    set(FLEET_CK_DIR "$ENV{FLEET_CK_DIR}")
endif()
if(NOT FLEET_CK_DIR)
    get_filename_component(
        _fleet_ws "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../aiter/3rdparty/composable_kernel_fleet" ABSOLUTE)
    if(EXISTS "${_fleet_ws}/include")
        set(FLEET_CK_DIR "${_fleet_ws}")
    endif()
endif()
if(NOT FLEET_CK_DIR OR NOT EXISTS "${FLEET_CK_DIR}/include")
    message(FATAL_ERROR "Fleet MPK provider build needs Composable Kernel headers. Set FLEET_CK_DIR to the Fleet "
                        "CK pin: a checkout of ROCm/composable_kernel at d8ee107a47d8485dbcffc79eb08e4f7c39ea6335.")
endif()

set(_r ${FLEET_MPK_ROCM_ROOT})
set(_inc ${CMAKE_CURRENT_SOURCE_DIR}/include/third_party/fleet/include)
set(_json ${CMAKE_CURRENT_SOURCE_DIR}/include/third_party/nlohmann/include)
set(_exports ${CMAKE_CURRENT_SOURCE_DIR}/provider.exports)
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/SOURCE_CLOSURE.txt _closure REGEX "^[^#]")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/SOURCE_CLOSURE.txt)
list(TRANSFORM _closure PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/)
set(_temps "")
if(FLEET_MPK_SAVE_TEMPS)
    set(_temps --save-temps)
endif()
set(_quiet "")
if(FLEET_MPK_QUIET_FWDPASS)
    set(_quiet -DMPK_QUIET_FWDPASS)
endif()

foreach(_cfg IN LISTS FLEET_MPK_CONFIGS)
    set(_dir ${FLEET_MPK_CONFIG_ROOT}/${_cfg})
    if(NOT EXISTS ${_dir}/kernel.hip)
        message(FATAL_ERROR "fleet MPK config ${_cfg}: ${_dir}/kernel.hip not found")
    endif()
    set(_bin ${CMAKE_CURRENT_BINARY_DIR}/${_cfg})
    file(MAKE_DIRECTORY ${_bin})
    file(STRINGS ${_dir}/compile_flags.txt _flags)
    # Read at configure time: an edit must re-configure, not rebuild with the flags of the last configure.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_dir}/compile_flags.txt)
    set(_so ${CMAKE_CURRENT_BINARY_DIR}/libaitermk_provider_gpt_oss_gfx950_fleet_mpk_${_cfg}.so)
    add_custom_command(
        OUTPUT ${_bin}/task_graph_blob.inc
        COMMAND ${CMAKE_COMMAND} -DIN=${_dir}/task_graph.json -DOUT=${_bin}/task_graph_blob.inc
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/task_graph_blob.cmake
        DEPENDS ${_dir}/task_graph.json ${CMAKE_CURRENT_SOURCE_DIR}/cmake/task_graph_blob.cmake
        VERBATIM)
    add_custom_command(
        OUTPUT ${_so}
        COMMAND ${CMAKE_COMMAND} -E env HIP_PATH=${_r} ROCM_PATH=${_r} HIP_CLANG_PATH=${_r}/llvm/bin
                HIP_DEVICE_LIB_PATH=${_r}/amdgcn/bitcode
                ${_r}/bin/hipcc -x hip ${_dir}/kernel.hip -I${_bin} -I${_inc} -I${_inc}/mirage/hip_compat
                -I${_inc}/mirage/persistent_kernel -I${FLEET_CK_DIR}/include -I${_r}/include -I${_json}
                ${_flags} ${_quiet} ${_temps} -shared -fPIC -Wl,--version-script=${_exports} -o ${_so}
        DEPENDS ${_dir}/kernel.hip ${_dir}/compile_flags.txt ${_bin}/task_graph_blob.inc ${_exports} ${_closure}
        WORKING_DIRECTORY ${_bin}
        COMMENT "fleet MPK provider ${_cfg}"
        COMMAND_EXPAND_LISTS
        VERBATIM)
    add_custom_target(aitermk_provider_gpt_oss_gfx950_fleet_mpk_${_cfg} ALL DEPENDS ${_so})
    install(FILES ${_so} DESTINATION ${CMAKE_INSTALL_LIBDIR})
endforeach()

