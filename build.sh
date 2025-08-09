rm -f ./tri ./main.o 
glslangValidator -V shaders/tri.vert.glsl -o shaders/tri.vert.spv
glslangValidator -V shaders/grid.vert.glsl -o shaders/grid.vert.spv
glslangValidator -V shaders/grid.frag.glsl -o shaders/grid.frag.spv
glslangValidator -V shaders/tri.frag.glsl -o shaders/tri.frag.spv
glslangValidator -V shaders/grass.frag.glsl -o shaders/grass.frag.spv
glslangValidator -V shaders/grass.vert.glsl -o shaders/grass.vert.spv

# clang -Wpointer-arith -Wformat=2  -Wall -Wextra -Wshadow -ggdb -std=c99 -pedantic main.c -o tri -D_DEBUG -DVK_USE_PLATFORM_WAYLAND_KHR -lvulkan -lm -lglfw  -lpthread -ldl && ./tri 
#  for tracy

mkdir -p build

if [ ! -f build/tracy_client.o ]; then
    echo "Compiling TracyClient.cpp..."
    clang++ -c ./external/tracy/public/TracyClient.cpp -o build/tracy_client.o -DTRACY_ENABLE -std=c++11 -O3  -march=native -lpthread -ldl 
fi

echo "Compiling main.c..."
clang++ -c main.c -o build/main.o \
    -Wpointer-arith -Wformat=2 -Wall -Wextra -Wshadow \
    -ggdb -pedantic \
    -D_DEBUG -DVK_USE_PLATFORM_WAYLAND_KHR -DTRACY_ENABLE

echo "Linking..."
clang++ build/main.o build/tracy_client.o -o build/tri \
    -lvulkan -lm -lglfw -lpthread -ldl


./build/tri

#

# gcc -ggdb lol2.c -o tri -D_DEBUG -DVK_USE_PLATFORM_XLIB_KHR -lvulkan -lglfw -lm && ./tri 
# -save-temps is very useful for preprocessed code and insights
