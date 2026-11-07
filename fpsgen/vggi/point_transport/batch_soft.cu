#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

// Cost storage may be reduced precision. Potentials, logsumexp and reductions
// are always FP32; there is no half-precision spatial search or hard output.
template<typename T,typename I> __global__ void dual_kernel(const int64_t* offsets,const I* neighbors,const T* cost,
 const float* other,float* result,int n,float eps,float theta){
 int tid=blockIdx.x*blockDim.x+threadIdx.x,i=tid/32,lane=tid%32;if(i>=n)return;
 int64_t a=offsets[i],b=offsets[i+1];float mx=-INFINITY;
 for(int64_t e=a+lane;e<b;e+=32)mx=fmaxf(mx,(other[neighbors[e]]-float(cost[e]))/eps);
 for(int d=16;d;d/=2)mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,d));
 mx=__shfl_sync(0xffffffff,mx,0);float sum=0;
 for(int64_t e=a+lane;e<b;e+=32)sum+=expf((other[neighbors[e]]-float(cost[e]))/eps-mx);
 for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);
 if(lane==0)result[i]=(-eps*(mx+logf(sum)))*theta;
}
void check_graph(torch::Tensor offsets,torch::Tensor neighbors,torch::Tensor cost,torch::Tensor f,torch::Tensor g){
 TORCH_CHECK(f.is_cuda() && f.scalar_type()==torch::kFloat32 && g.scalar_type()==torch::kFloat32 && f.dim()==1 && g.sizes()==f.sizes() && f.is_contiguous() && g.is_contiguous(),"FP32 potential vectors required");
 TORCH_CHECK(offsets.scalar_type()==torch::kInt64 && (neighbors.scalar_type()==torch::kInt64 || neighbors.scalar_type()==torch::kInt32) && offsets.numel()==f.numel()+1 && neighbors.numel()==cost.numel() && offsets.is_contiguous() && neighbors.is_contiguous() && cost.is_contiguous(),"Invalid CSR");
 TORCH_CHECK(offsets.device()==f.device() && neighbors.device()==f.device() && cost.device()==f.device() && g.device()==f.device(),"Device mismatch");
 TORCH_CHECK(cost.scalar_type()==torch::kFloat32 || cost.scalar_type()==torch::kFloat16 || cost.scalar_type()==torch::kBFloat16,"FP32/FP16/BF16 costs required");
}
void dual(torch::Tensor offsets,torch::Tensor neighbors,torch::Tensor cost,torch::Tensor other,torch::Tensor result,double eps,double theta){
 check_graph(offsets,neighbors,cost,other,result);TORCH_CHECK(eps>0 && theta>0 && theta<=1,"Invalid dual parameters");
 c10::cuda::CUDAGuard guard(result.device());auto stream=at::cuda::getCurrentCUDAStream();int n=result.numel();
 #define DUAL(I) dual_kernel<scalar_t,I><<<(n+3)/4,128,0,stream>>>(offsets.data_ptr<int64_t>(),neighbors.data_ptr<I>(),cost.data_ptr<scalar_t>(),other.data_ptr<float>(),result.data_ptr<float>(),n,eps,theta)
 AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,cost.scalar_type(),"batch_soft_dual",[&]{
 if(neighbors.scalar_type()==torch::kInt32){DUAL(int32_t);}else{DUAL(int64_t);}
 });
 #undef DUAL
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename T,typename I> __global__ void moments_kernel(const int64_t* offsets,const I* neighbors,const T* costs,
 const float* f,const float* g,const float* x,const float* y,float* out,int n,float eps){
 int tid=blockIdx.x*blockDim.x+threadIdx.x,i=tid/32,lane=tid%32;if(i>=n)return;
 float v[8]={0,0,0,0,0,0,0,0};
 for(int64_t e=offsets[i]+lane;e<offsets[i+1];e+=32){
  int j=neighbors[e];float c=float(costs[e]),logq=((f[i]+g[j])-c)/eps,q=expf(logq);
  v[0]+=q;v[1]+=q*(x[3*i]-y[3*j]);v[2]+=q*(x[3*i+1]-y[3*j+1]);v[3]+=q*(x[3*i+2]-y[3*j+2]);
  v[4]+=q*c;v[5]+=q*logq;v[6]+=q*q;v[7]=fmaxf(v[7],q);
 }
 for(int d=16;d;d/=2){
  #pragma unroll
  for(int k=0;k<7;k++)v[k]+=__shfl_down_sync(0xffffffff,v[k],d);
  v[7]=fmaxf(v[7],__shfl_down_sync(0xffffffff,v[7],d));
 }
 if(lane==0){for(int k=0;k<8;k++)out[int64_t(i)*8+k]=v[k];}
}
torch::Tensor moments(torch::Tensor offsets,torch::Tensor neighbors,torch::Tensor cost,torch::Tensor f,torch::Tensor g,torch::Tensor x,torch::Tensor y,double eps){
 check_graph(offsets,neighbors,cost,f,g);
 TORCH_CHECK(x.device()==f.device() && y.device()==f.device() && x.scalar_type()==torch::kFloat32 && y.scalar_type()==torch::kFloat32 && x.is_contiguous() && y.is_contiguous() && x.dim()==2 && x.size(1)==3 && y.sizes()==x.sizes() && x.size(0)==f.numel() && eps>0,"Invalid moment inputs");
 c10::cuda::CUDAGuard guard(f.device());auto stream=at::cuda::getCurrentCUDAStream();int n=f.numel();auto out=torch::empty({n,8},f.options());
 #define MOMENTS(I) moments_kernel<scalar_t,I><<<(n+3)/4,128,0,stream>>>(offsets.data_ptr<int64_t>(),neighbors.data_ptr<I>(),cost.data_ptr<scalar_t>(),f.data_ptr<float>(),g.data_ptr<float>(),x.data_ptr<float>(),y.data_ptr<float>(),out.data_ptr<float>(),n,eps)
 AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,cost.scalar_type(),"batch_soft_moments",[&]{
 if(neighbors.scalar_type()==torch::kInt32){MOMENTS(int32_t);}else{MOMENTS(int64_t);}
 });
 #undef MOMENTS
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

__global__ void keys_kernel(const int64_t* forward,const int64_t* reverse,int64_t* keys,int64_t count,int n,int k,int64_t total){
 int64_t e=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(e>=count)return;
 int64_t row=e/k,base=(row/n)*n;
 keys[e]=row*total+base+forward[e];keys[count+e]=(base+reverse[e])*total+row;
}
torch::Tensor keys(torch::Tensor forward,torch::Tensor reverse){
 TORCH_CHECK(forward.is_cuda() && reverse.device()==forward.device() && forward.scalar_type()==torch::kInt64 && reverse.scalar_type()==torch::kInt64 && forward.is_contiguous() && reverse.is_contiguous() && forward.dim()==3 && forward.sizes()==reverse.sizes() && forward.numel()>0,"Invalid batched KNN IDs");
 c10::cuda::CUDAGuard guard(forward.device());auto stream=at::cuda::getCurrentCUDAStream();
 int64_t count=forward.numel(),total=forward.size(0)*forward.size(1);auto out=torch::empty({2*count},forward.options());
 keys_kernel<<<(count+255)/256,256,0,stream>>>(forward.data_ptr<int64_t>(),reverse.data_ptr<int64_t>(),out.data_ptr<int64_t>(),count,forward.size(1),forward.size(2),total);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("dual",&dual);m.def("moments",&moments);m.def("keys",&keys);}
