cmake_minimum_required(VERSION 3.7 FATAL_ERROR)
project(va)

set (CMAKE_CXX_STANDARD 17)

include(ExternalProject)

if(WIN32)
    #get GLFW and add it to our build
    include(FetchContent)
    FetchContent_Declare(
      GLFW
      GIT_REPOSITORY https://github.com/glfw/glfw.git
      GIT_TAG        3.3.2
    )
  FetchContent_MakeAvailable(GLFW)

  #get glm and make it available to include
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

  set(FETCHED_DEPS "glfw glm")
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GLFW REQUIRED glfw3)
  include_directories(${GLFW_INCLUDE_DIR})
  pkg_check_modules(GLM REQUIRED glm)
  include_directories(${GLM_INCLUDE_DIR})

  # none needed since the package manager provides
  # glfw and glm
  set(FETCHED_DEPS "")
endif()


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
add_dependencies(va ${FETCHED_DEPS} stb tinyobjloader Vulkan::Vulkan)
if(WIN32)
  target_compile_definitions(va PRIVATE VK_USE_PLATFORM_WIN32_KHR)
else()
  # on Linux, need to explictly link to this even though C++17 is set
  set(STDC++FS "stdc++fs")
endif(WIN32)
target_link_libraries(va ${GLFW_LIBRARIES} ${GLM_LIBRARIES} ${STDC++FS} Vulkan::Vulkan )
target_include_directories(va PRIVATE ${GLFW_SOURCE_DIR}/include ${GLM_SOURCE_DIR}/include ${GlmIncludeDir} ${StbIncludeDir} ${Vulkan_INCLUDE_DIR} ${TinyobjloaderIncludeDir})

#add_custom_target(Shaders ALL SOURCES ${va_shaders} DEPENDS ${va_shaders})

set(SHADERS_DIR "${CMAKE_SOURCE_DIR}/shaders")
if(WIN32)
STRING(REGEX REPLACE "/" "\\\\" SHADERS_DIR ${SHADERS_DIR})
endif()
if(WIN32)
    configure_file(./shaders/CompileShaders.bat.in ${CMAKE_BINARY_DIR}/CompileShaders.bat @ONLY)

    add_custom_command(TARGET Shaders POST_BUILD DEPENDS ${va_shaders}
        COMMAND ${CMAKE_BINARY_DIR}/CompileShaders.bat
        )
else()
    configure_file(./shaders/CompileShaders.sh.in ${CMAKE_BINARY_DIR}/CompileShaders.sh @ONLY)

    add_custom_command(TARGET va POST_BUILD DEPENDS ${va_shaders}
        COMMAND ${CMAKE_BINARY_DIR}/CompileShaders.sh
        )
endif()


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
add_custom_command(TARGET va PRE_BUILD
    COMMAND cp -r "${TEXTURES_DIR}" "${OUTPUT_COPY_DIR}/" || :
)
add_custom_command(TARGET va PRE_BUILD
    COMMAND mkdir "${OUTPUT_COPY_DIR}/shaders" || :
    COMMAND cp "${SHADERS_DIR}/*.vert" "${SHADERS_DIR}/*.frag" "${OUTPUT_COPY_DIR}/shaders" || :
)
add_custom_command(TARGET va PRE_BUILD
      COMMAND cp -r "${MODELS_DIR}" "${OUTPUT_COPY_DIR}/models" || :
)
endif()
