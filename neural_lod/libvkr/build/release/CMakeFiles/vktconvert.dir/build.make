# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/phil/Documents/chameleonrt_dfki/neural_lod/libvkr

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/phil/Documents/chameleonrt_dfki/neural_lod/libvkr/build/release

# Include any dependencies generated for this target.
include CMakeFiles/vktconvert.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/vktconvert.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/vktconvert.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/vktconvert.dir/flags.make

CMakeFiles/vktconvert.dir/src/vktconvert.c.o: CMakeFiles/vktconvert.dir/flags.make
CMakeFiles/vktconvert.dir/src/vktconvert.c.o: /home/phil/Documents/chameleonrt_dfki/neural_lod/libvkr/src/vktconvert.c
CMakeFiles/vktconvert.dir/src/vktconvert.c.o: CMakeFiles/vktconvert.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/phil/Documents/chameleonrt_dfki/neural_lod/libvkr/build/release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/vktconvert.dir/src/vktconvert.c.o"
	/usr/bin/clang-9 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/vktconvert.dir/src/vktconvert.c.o -MF CMakeFiles/vktconvert.dir/src/vktconvert.c.o.d -o CMakeFiles/vktconvert.dir/src/vktconvert.c.o -c /home/phil/Documents/chameleonrt_dfki/neural_lod/libvkr/src/vktconvert.c

CMakeFiles/vktconvert.dir/src/vktconvert.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/vktconvert.dir/src/vktconvert.c.i"
	/usr/bin/clang-9 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/phil/Documents/chameleonrt_dfki/neural_lod/libvkr/src/vktconvert.c > CMakeFiles/vktconvert.dir/src/vktconvert.c.i

CMakeFiles/vktconvert.dir/src/vktconvert.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/vktconvert.dir/src/vktconvert.c.s"
	/usr/bin/clang-9 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/phil/Documents/chameleonrt_dfki/neural_lod/libvkr/src/vktconvert.c -o CMakeFiles/vktconvert.dir/src/vktconvert.c.s

# Object files for target vktconvert
vktconvert_OBJECTS = \
"CMakeFiles/vktconvert.dir/src/vktconvert.c.o"

# External object files for target vktconvert
vktconvert_EXTERNAL_OBJECTS =

vktconvert: CMakeFiles/vktconvert.dir/src/vktconvert.c.o
vktconvert: CMakeFiles/vktconvert.dir/build.make
vktconvert: libvkr.a
vktconvert: CMakeFiles/vktconvert.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/phil/Documents/chameleonrt_dfki/neural_lod/libvkr/build/release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable vktconvert"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vktconvert.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/vktconvert.dir/build: vktconvert
.PHONY : CMakeFiles/vktconvert.dir/build

CMakeFiles/vktconvert.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/vktconvert.dir/cmake_clean.cmake
.PHONY : CMakeFiles/vktconvert.dir/clean

CMakeFiles/vktconvert.dir/depend:
	cd /home/phil/Documents/chameleonrt_dfki/neural_lod/libvkr/build/release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/phil/Documents/chameleonrt_dfki/neural_lod/libvkr /home/phil/Documents/chameleonrt_dfki/neural_lod/libvkr /home/phil/Documents/chameleonrt_dfki/neural_lod/libvkr/build/release /home/phil/Documents/chameleonrt_dfki/neural_lod/libvkr/build/release /home/phil/Documents/chameleonrt_dfki/neural_lod/libvkr/build/release/CMakeFiles/vktconvert.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/vktconvert.dir/depend
