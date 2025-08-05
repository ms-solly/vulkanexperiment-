rm ./tri
glslangValidator -V shaders/tri.vert.glsl -o shaders/tri.vert.spv
glslangValidator -V shaders/grid.vert.glsl -o shaders/grid.vert.spv
glslangValidator -V shaders/grid.frag.glsl -o shaders/grid.frag.spv
glslangValidator -V shaders/tri.frag.glsl -o shaders/tri.frag.spv
glslangValidator -V shaders/grass.frag.glsl -o shaders/grass.frag.spv
glslangValidator -V shaders/grass.vert.glsl -o shaders/grass.vert.spv
clang -Wpointer-arith -Wformat=2  -Wall -Wextra -Wshadow -ggdb -std=c99 -pedantic main.c -o tri -D_DEBUG -DVK_USE_PLATFORM_WAYLAND_KHR -lvulkan -lm -lglfw  -lpthread -ldl && ./tri 
# gcc -ggdb lol2.c -o tri -D_DEBUG -DVK_USE_PLATFORM_XLIB_KHR -lvulkan -lglfw -lm && ./tri 
# -save-temps is very useful for preprocessed code and insights 