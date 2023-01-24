
# Add source to this project's executable.
add_executable(vulkan-project
    main.cpp
    app.cpp
    app.h
    vk_types.h)


set_property(TARGET vulkan-project PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:vulkan-project>")

target_include_directories(vulkan-project PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(vulkan-project vkbootstrap vma glm)

target_link_libraries(vulkan-project Vulkan::Vulkan sdl2)

add_dependencies(vulkan-project Shaders)
