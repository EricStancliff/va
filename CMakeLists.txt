cmake_minimum_required(VERSION 3.7 FATAL_ERROR)
project(va)

set (CMAKE_CXX_STANDARD 17)

#get GLFW and add it to our build
include(FetchContent)
FetchContent_Declare(
  GLFW
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG        3.3.2
)
FetchContent_MakeAvailable(GLFW)

#get glm and make it available to include
include(ExternalProject)
ExternalProject_Add(
    glm
    PREFIX ${CMAKE_BINARY_DIR}/glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
	GIT_TAG        0.9.9.7
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
    )
ExternalProject_Get_Property(glm source_dir)
set(GlmIncludeDir ${source_dir})

#get stb and make it available to include
ExternalProject_Add(
    stb
    PREFIX ${CMAKE_BINARY_DIR}/stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
	GIT_TAG        origin/master
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
    )
ExternalProject_Get_Property(stb source_dir)
set(StbIncludeDir ${source_dir})

#get tinyobjloader and make it available to include
ExternalProject_Add(
    tinyobjloader
    PREFIX ${CMAKE_BINARY_DIR}/tinyobjloader
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
	GIT_TAG        origin/master
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
    )
ExternalProject_Get_Property(tinyobjloader source_dir)
set(TinyobjloaderIncludeDir ${source_dir})

find_package(Vulkan REQUIRED)

set(va_includes
	./core/VulkanInterfacePrivate.h
    ./core/EventState.h
    ./core/VulkanInterface.h
)

set(va_source
    main.cpp
	./core/VulkanInterfacePrivate.cpp
    ./core/VulkanInterface.cpp
)

set(va_shaders
	./shaders/SimpleTriangle.vert
	./shaders/SimpleTriangle.frag
)

add_executable(va ${va_includes} ${va_source})
add_dependencies(va glfw glm stb Vulkan::Vulkan)
target_compile_definitions(va PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_link_libraries(va glfw Vulkan::Vulkan)
target_include_directories(va PRIVATE ${GLFW_SOURCE_DIR}/include ${GlmIncludeDir} ${StbIncludeDir} ${Vulkan_INCLUDE_DIR} ${TinyobjloaderIncludeDir})

add_custom_target(Shaders ALL SOURCES ${va_shaders} DEPENDS ${va_shaders})

set(SHADERS_DIR "${CMAKE_SOURCE_DIR}/shaders")
if(WIN32)
STRING(REGEX REPLACE "/" "\\\\" SHADERS_DIR ${SHADERS_DIR})
endif()

configure_file(./shaders/CompileShaders.bat.in ${CMAKE_BINARY_DIR}/CompileShaders.bat @ONLY) 

add_custom_command(TARGET Shaders POST_BUILD DEPENDS ${va_shaders}
    COMMAND ${CMAKE_BINARY_DIR}/CompileShaders.bat
)

set(TEXTURES_DIR ${CMAKE_SOURCE_DIR}/textures)
set(SHADERS_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(MODELS_DIR ${CMAKE_SOURCE_DIR}/models)
set(OUTPUT_COPY_DIR ${CMAKE_CURRENT_BINARY_DIR})
if(WIN32)
STRING(REGEX REPLACE "/" "\\\\" TEXTURES_DIR ${TEXTURES_DIR})
STRING(REGEX REPLACE "/" "\\\\" SHADERS_DIR ${SHADERS_DIR})
STRING(REGEX REPLACE "/" "\\\\" MODELS_DIR ${MODELS_DIR})
STRING(REGEX REPLACE "/" "\\\\" OUTPUT_COPY_DIR ${OUTPUT_COPY_DIR})
add_custom_command(TARGET va POST_BUILD
    COMMAND xcopy /i /s /y "${TEXTURES_DIR}" "${OUTPUT_COPY_DIR}\\$<CONFIGURATION>\\textures"
)
add_custom_command(TARGET va POST_BUILD
    COMMAND xcopy /i /s /y "${SHADERS_DIR}\\*.spv" "${OUTPUT_COPY_DIR}\\$<CONFIGURATION>\\shaders"
)
add_custom_command(TARGET va POST_BUILD
    COMMAND xcopy /i /s /y "${MODELS_DIR}" "${OUTPUT_COPY_DIR}\\$<CONFIGURATION>\\models"
)
set_target_properties(va PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:va>"
                                                   VS_DEBUGGER_COMMAND           "$<TARGET_FILE:va>"
                                                   VS_DEBUGGER_ENVIRONMENT       "PATH=%PATH%;${CMAKE_PREFIX_PATH}/bin")
else()
add_custom_command(TARGET va POST_BUILD
    COMMAND cp "${TEXTURES_DIR}" "${OUTPUT_COPY_DIR}/textures"
)
add_custom_command(TARGET va POST_BUILD
    COMMAND cp "${SHADERS_DIR}/*.spv" "${OUTPUT_COPY_DIR}/shaders"
)
add_custom_command(TARGET va POST_BUILD
    COMMAND cp "${MODELS_DIR}" "${OUTPUT_COPY_DIR}/models"
)
endif()