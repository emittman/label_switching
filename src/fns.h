#include "curand_kernel.h"
#include "curand.h"
#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include<thrust/copy.h>
#include<thrust/sequence.h>
#include<thrust/sort.h>
#include<curand.h>

typedef thrust::device_vector<double> fvec_d;
typedef thrust::device_vector<int> ivec_d;
typedef thrust::host_vector<int> ivec_h;


__global__ void setup_kernel(int seed, int n_threads, curandState *states);
__global__ void getExponential(curandState *states, int n_threads, double *weights, double *result);
void sample_wwr(curandState *states, fvec_d &weights, ivec_d &result);
extern "C" SEXP Rsample_wwr(SEXP Rseed, SEXP Rweights);