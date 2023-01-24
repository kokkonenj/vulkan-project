Requirements:
  - VulkanSDK
  - SDL2

Code is developed on Windows and Visual Studio, but it should work cross-platform.

Building the program:
  1. Open the CMakeLists.txt on the root of the repository with CMake, and change sdl2_DIR variable to the root of your SDL2 library folder
  2. Configure and generate project files
  3. Open project in Visual Studio, build & run vulkan-project

If you get an error saying "missing SDL2.dll", go to your SDL2 folder and copy "lib/x64/SDL2.dll" to "vulkan-project/bin/Debug(or Release)" where
"vulcan-project.exe" should be. Re-run and program should now work. If not, check that you are building for x64.
