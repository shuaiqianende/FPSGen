#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
// Exact union of directed KNN graphs. Sorted row IDs permit O(log K)
// duplicate detection, avoiding a global sort of 2BNK 64-bit edge keys.
__global__ void count_edges(const int64_t* f,const int64_t* r,uint8_t* missing,int* rows,int* cols,int n,int k,int64_t edges){
 int64_t e=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(e>=edges)return;
 int row=e/k,base=row/n*n,j=base+f[e],i=base+r[e];
 atomicAdd(cols+j,1);
 int lo=0,hi=k;const int64_t* a=f+int64_t(i)*k;int target=row-base;
 while(lo<hi){int mid=(lo+hi)/2;if(a[mid]<target)lo=mid+1;else hi=mid;}
 bool add=lo==k||a[lo]!=target;missing[e]=add;
 if(add){atomicAdd(rows+i,1);atomicAdd(cols+row,1);}
}
__device__ void put(int i,int j,int64_t pos,const float* x,const float* y,const int64_t* co,int* cc,int* t,float* c,int* s,float* ct){
 float dx=x[3*i]-y[3*j],dy=x[3*i+1]-y[3*j+1],dz=x[3*i+2]-y[3*j+2];float v=.5f*((dx*dx+dz*dz)+dy*dy);
 t[pos]=j;c[pos]=v;int64_t rev=co[j]+atomicAdd(cc+j,1);s[rev]=i;ct[rev]=v;
}
__global__ void fill_edges(const int64_t* f,const int64_t* r,const uint8_t* missing,const float* x,const float* y,const int64_t* ro,const int64_t* co,int* rc,int* cc,int* t,float* c,int* s,float* ct,int n,int k,int64_t edges){
 int64_t e=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(e>=edges)return;
 int row=e/k,base=row/n*n;
 put(row,base+f[e],ro[row]+e%k,x,y,co,cc,t,c,s,ct);
 if(missing[e]){int i=base+r[e];put(i,row,ro[i]+k+atomicAdd(rc+i,1),x,y,co,cc,t,c,s,ct);}
}
std::vector<torch::Tensor> graph(torch::Tensor f,torch::Tensor r,torch::Tensor x,torch::Tensor y){
 TORCH_CHECK(f.is_cuda()&&f.scalar_type()==torch::kInt64&&r.scalar_type()==torch::kInt64&&f.dim()==3&&f.sizes()==r.sizes()&&f.is_contiguous()&&r.is_contiguous(),"KNN tensors required");
 TORCH_CHECK(x.device()==f.device()&&y.device()==f.device()&&r.device()==f.device()&&x.scalar_type()==torch::kFloat32&&y.scalar_type()==torch::kFloat32&&x.is_contiguous()&&y.is_contiguous()&&x.dim()==3&&x.size(2)==3&&x.sizes()==y.sizes()&&x.size(0)==f.size(0)&&x.size(1)==f.size(1),"Invalid coordinates");
 c10::cuda::CUDAGuard guard(f.device());auto stream=at::cuda::getCurrentCUDAStream();
 int n=f.size(1),k=f.size(2),total=f.size(0)*n;int64_t edges=f.numel();
 auto io=f.options().dtype(torch::kInt32);auto rows=torch::full({total},k,io),cols=torch::zeros({total},io),flags=torch::empty({edges},f.options().dtype(torch::kUInt8));
 count_edges<<<(edges+255)/256,256,0,stream>>>(f.data_ptr<int64_t>(),r.data_ptr<int64_t>(),flags.data_ptr<uint8_t>(),rows.data_ptr<int>(),cols.data_ptr<int>(),n,k,edges);
 auto ro=torch::cat({torch::zeros({1},f.options()),rows.cumsum(0,torch::kInt64)}),co=torch::cat({torch::zeros({1},f.options()),cols.cumsum(0,torch::kInt64)});
 auto rc=torch::zeros({total},io),cc=torch::zeros({total},io),t=torch::empty({2*edges},io),s=torch::empty_like(t),c=torch::empty({2*edges},x.options()),ct=torch::empty_like(c);
 fill_edges<<<(edges+255)/256,256,0,stream>>>(f.data_ptr<int64_t>(),r.data_ptr<int64_t>(),flags.data_ptr<uint8_t>(),x.data_ptr<float>(),y.data_ptr<float>(),ro.data_ptr<int64_t>(),co.data_ptr<int64_t>(),rc.data_ptr<int>(),cc.data_ptr<int>(),t.data_ptr<int>(),c.data_ptr<float>(),s.data_ptr<int>(),ct.data_ptr<float>(),n,k,edges);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {ro,co,t,c,s,ct};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("graph",&graph);}
