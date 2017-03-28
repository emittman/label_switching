#include<fns.h>
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include<thrust/copy.h>
#include<thrust/sequence.h>
#include<thrust/sort.h>
#include<curand.h>

typedef thrust::device_vector<double> fvec_d;
typedef thrust::device_vector<int> ivec_d;
typedef thrust::host_vector<int> ivec_d;

__global__ void setup_kernel(int seed, int n_threads, curandState *states) {
  
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id < n_threads){
  /* Each thread gets same seed, a different sequence number, no offset */
    curand_init(seed, id, 0, &states[id]);
  }
}

__global__ void getExponential(curandState *states, int n_threads, double *weights, double *result){
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id < n_threads){
    result[id] = -log(curand_uniform(&(states[id]))) / weights[id];
  }
}

void sample_wwr(curandState *states, fvec_d &weights, ivec_d &result){
  unsigned N = weights.size();
  fvec_d e(N);
  thrust::sequence(result.begin(), result.end());
  unsigned blocksize = 512;
  unsigned nblocks = N/512 + 1;
  getExponential<<<nblocks, blocksize>>>(states, N, thrust::raw_pointer_cast(weights.data()), thrust::raw_pointer_cast(e.data()));
  thrust::sort_by_key(e.begin(), e.end(), result.begin());
}

extern "C" SEXP Rsample_wwr(SEXP Rseed, SEXP Rweights){
  int N = length(Rweights), seed = INTEGER(Rseed)[0];
  fvec_d weights(REAL(Rweights), REAL(Rweights) + N);
  ivec_d out_d(N);
  //instantiate RNGs
  curandState *devStates;
  CUDA_CALL(cudaMalloc((void **) &devStates, N * sizeof(curandState)));
  unsigned blocksize = 512;
  unsigned nblocks = N/512 + 1;
  setup_kernel<<<nblocks, blocksize>>>(seed, N, devStates);
  
  sample_wwr(devStates, weights, out_d);
  
  thrust::host_vector<int> out_h(N);
  thrust::copy(out_d.begin(), out_d.end(), out_h.begin());
  SEXP out = PROTECT(allocVector(INTSXP, N));
  for(int i=0; i<N; i++) INTEGER(out)[i] = out_h[i];
  UNPROTECT(1);
  cudaFree(devStates);
  return out;
}

