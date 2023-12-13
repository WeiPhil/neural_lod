#pragma once

// Standard libs
#include <stdio.h>
#include <cstdlib>
// Cuda
#include "cuda_runtime.h"
#include "libs/cuda/helper_cuda.h"

// Function to check cuda requirements
bool initCuda();