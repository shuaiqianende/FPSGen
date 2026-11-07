#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

__global__ void decode_kernel(const int64_t* keys,int64_t edges,int64_t n,
 const float* x,const float* y,int64_t* sources,int64_t* targets,float* costs){
 int64_t e=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;if(e>=edges)return;
 int64_t s=keys[e]/n,t=keys[e]-s*n;sources[e]=s;targets[e]=t;
 float dx=x[3*s]-y[3*t],dy=x[3*s+1]-y[3*t+1],dz=x[3*s+2]-y[3*t+2];
 // Match ATen's three-element CUDA reduction tree; do not fuse multiply-add.
 costs[e]=.5f*((dx*dx+dz*dz)+dy*dy);
}

void decode(torch::Tensor keys,torch::Tensor x,torch::Tensor y,
 torch::Tensor sources,torch::Tensor targets,torch::Tensor costs){
 TORCH_CHECK(keys.is_cuda() && keys.scalar_type()==torch::kInt64 && keys.is_contiguous()
  && x.scalar_type()==torch::kFloat32 && y.scalar_type()==torch::kFloat32
  && x.is_contiguous() && y.is_contiguous() && x.size(0)==y.size(0),"Invalid decode input");
 c10::cuda::CUDAGuard guard(keys.device());auto stream=at::cuda::getCurrentCUDAStream();
 decode_kernel<<<(keys.numel()+255)/256,256,0,stream>>>(keys.data_ptr<int64_t>(),keys.numel(),x.size(0),
 x.data_ptr<float>(),y.data_ptr<float>(),sources.data_ptr<int64_t>(),targets.data_ptr<int64_t>(),costs.data_ptr<float>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Counting-scatter transpose: identical graph edges, no global comparison sort.
// Within-column ordering is intentionally unspecified (FP32 reductions may differ).
__global__ void transpose_kernel(const int64_t* sources,const int64_t* targets,
 const float* costs,const int64_t* offsets,int* cursors,int64_t edges,
 int64_t* reverse_sources,float* reverse_costs){
 int64_t e=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;if(e>=edges)return;
 int64_t j=targets[e];int slot=atomicAdd(cursors+j,1);
 int64_t out=offsets[j]+slot;reverse_sources[out]=sources[e];reverse_costs[out]=costs[e];
}
void transpose(torch::Tensor s,torch::Tensor t,torch::Tensor c,torch::Tensor offsets,
 torch::Tensor cursors,torch::Tensor reverse_sources,torch::Tensor reverse_costs){
 TORCH_CHECK(s.is_cuda() && s.scalar_type()==torch::kInt64 && t.scalar_type()==torch::kInt64
             && c.scalar_type()==torch::kFloat32 && offsets.scalar_type()==torch::kInt64
             && cursors.scalar_type()==torch::kInt32,"Invalid graph tensors");
 TORCH_CHECK(s.is_contiguous() && t.is_contiguous() && c.is_contiguous()
             && s.numel()==t.numel() && s.numel()==c.numel(),"Invalid graph shape");
 c10::cuda::CUDAGuard guard(s.device());auto stream=at::cuda::getCurrentCUDAStream();
 transpose_kernel<<<(s.numel()+255)/256,256,0,stream>>>(s.data_ptr<int64_t>(),t.data_ptr<int64_t>(),
 c.data_ptr<float>(),offsets.data_ptr<int64_t>(),cursors.data_ptr<int>(),s.numel(),
 reverse_sources.data_ptr<int64_t>(),reverse_costs.data_ptr<float>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("transpose",&transpose);m.def("decode",&decode);}
