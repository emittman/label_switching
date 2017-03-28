#include "curand_kernel.h"
#include "curand.h"
#include <R.h>

__global__ void setup_kernel(int seed, int n_threads, curandState *states);
__global__ void getExponential(curandState *states, int n_threads, double *weights, double *result);
void sample_wwr(curandState *states, fvec_d &weights, ivec_d &result);
extern "C" SEXP Rsample_wwr(SEXP Rseed, SEXP Rweights);