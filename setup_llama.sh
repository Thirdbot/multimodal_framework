#!/bin/bash

set -e

# Clone or update llama.cpp
if [ ! -d "third_party/llama.cpp" ]; then
  echo "Cloning llama.cpp..."
  git submodule add https://github.com/ggerganov/llama.cpp Llama/llama.cpp
else
  echo "Updating llama.cpp..."
  cd third_party/llama.cpp
  git pull origin master
  cd ../../
fi

# Optional: Build llama.cpp
echo "Building llama.cpp..."
cd third_party/llama.cpp
make -j$(nproc)
cd ../../

echo "llama.cpp setup complete!"
