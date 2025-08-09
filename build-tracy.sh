if [ ! -d "build/profiler" ]; then
    cmake -B build/profiler external/tracy/profiler 
    cmake --build build/profiler --config Release
fi
# for launching 
./build/profiler/tracy-profiler 