#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <curand_kernel.h>
__device__ float uniform(curandStatePhilox4_32_10_t* r){return (curand(r)>>8)*0x1p-24f;}
__device__ int choose(const float* c,int n,curandStatePhilox4_32_10_t* r){
 float v=fminf(uniform(r)*c[n-1],nextafterf(c[n-1],0.f));int lo=0,hi=n-1;
 while(lo<hi){int m=(lo+hi)/2;if(c[m]<=v)lo=m+1;else hi=m;}return lo;
}
__device__ float basis(float x){x=fabsf(x);if(x<1)return 2.f/3-x*x+.5f*x*x*x;if(x<2){float t=2-x;return t*t*t/6;}return 0;}
__global__ void sample_kernel(const float* a,const float* upper,const float* c0,const float* c1,const float* c2,const float* origin,const float* h,float* out,int* tries,int* errors,int nx,int ny,int nz,int ntop,int count,int limit,uint64_t seed){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=count)return;
 curandStatePhilox4_32_10_t rng;curand_init(seed,id,0,&rng);
 for(int attempt=1;attempt<=limit;attempt++){
  int top=choose(c2,ntop,&rng),group=top*256+choose(c1+top*256,256,&rng),cell=group*256+choose(c0+group*256,256,&rng);
  int ix=cell/(ny*nz),iy=(cell/nz)%ny,iz=cell%nz;
  float u[3]={ix+uniform(&rng)-.5f,iy+uniform(&rng)-.5f,iz+uniform(&rng)-.5f};
  int lower[3];float b[3][4];
  for(int d=0;d<3;d++){lower[d]=int(floorf(u[d]))-1;for(int j=0;j<4;j++)b[d][j]=basis(u[d]-lower[d]-j);}
  float amp=0;
  for(int i=0;i<4;i++){int x=lower[0]+i;if(x<0||x>=nx)continue;
   for(int j=0;j<4;j++){int y=lower[1]+j;if(y<0||y>=ny)continue;float xy=b[0][i]*b[1][j];
    for(int k=0;k<4;k++){int z=lower[2]+k;if(z>=0&&z<nz)amp+=a[(x*ny+y)*nz+z]*xy*b[2][k];}}}
  float rho=amp*amp;
  if(!isfinite(rho)||rho>upper[cell]*(1+1e-5f)){atomicAdd(errors,1);tries[id]=attempt;return;}
  if(uniform(&rng)*upper[cell]<rho){
   for(int d=0;d<3;d++){int shape=d==0?nx:(d==1?ny:nz);float hi=origin[d]+shape*h[d];out[3*id+d]=fminf(origin[d]+(u[d]+.5f)*h[d],nextafterf(hi,origin[d]));}
   tries[id]=attempt;return;
  }
 }
 tries[id]=limit;atomicAdd(errors,1);
}
std::vector<torch::Tensor> sample(torch::Tensor a,torch::Tensor upper,torch::Tensor c0,torch::Tensor c1,torch::Tensor c2,torch::Tensor origin,torch::Tensor h,int64_t count,int64_t seed,int64_t limit){
 TORCH_CHECK(a.is_cuda()&&a.dim()==3&&a.scalar_type()==torch::kFloat32&&a.is_contiguous()&&count>0&&count<2147483647&&limit>0&&limit<2147483647,"Invalid sampler input");
 for(auto t:{upper,c0,c1,c2,origin,h})TORCH_CHECK(t.device()==a.device()&&t.scalar_type()==torch::kFloat32&&t.is_contiguous(),"Sampler tensor mismatch");
 TORCH_CHECK(origin.numel()==3&&h.numel()==3&&upper.numel()==a.numel()&&c0.numel()>=a.numel()&&c0.numel()%256==0&&c1.numel()>=c0.numel()/256&&c1.numel()%256==0&&c2.numel()==c1.numel()/256,"Invalid CDF geometry");
 c10::cuda::CUDAGuard guard(a.device());auto out=torch::empty({count,3},a.options()),tries=torch::empty({count},a.options().dtype(torch::kInt32)),errors=torch::zeros({1},tries.options());
 sample_kernel<<<(count+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(a.data_ptr<float>(),upper.data_ptr<float>(),c0.data_ptr<float>(),c1.data_ptr<float>(),c2.data_ptr<float>(),origin.data_ptr<float>(),h.data_ptr<float>(),out.data_ptr<float>(),tries.data_ptr<int>(),errors.data_ptr<int>(),a.size(0),a.size(1),a.size(2),c2.numel(),count,limit,seed);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,tries,errors};
}
// Banded triangular Gram inverse: one independent line per thread.
__global__ void decode_axis(float* q,const float* r,int nx,int ny,int nz,int axis){
 int n=axis==0?nx:(axis==1?ny:nz),lines=nx*ny*nz/n;
 int line=blockIdx.x*blockDim.x+threadIdx.x;if(line>=lines)return;
 int stride=axis==0?ny*nz:(axis==1?nz:1);
 int base=axis==0?line:(axis==1?(line/nz)*ny*nz+line%nz:line*nz);
 for(int j=n-1;j>=0;j--){float v=q[base+j*stride];for(int d=1;d<=3&&j+d<n;d++)v-=r[j*n+j+d]*q[base+(j+d)*stride];q[base+j*stride]=v/r[j*n+j];}
}
torch::Tensor decode(torch::Tensor q,std::vector<torch::Tensor> factors,double scale){
 TORCH_CHECK(q.is_cuda()&&q.dim()==3&&q.scalar_type()==torch::kFloat32&&q.is_contiguous()&&factors.size()==3&&scale>0,"Invalid Gram state");
 c10::cuda::CUDAGuard guard(q.device());auto out=q*float(sqrt(scale));
 for(int axis=0;axis<3;axis++){auto r=factors[axis];int n=q.size(axis);TORCH_CHECK(r.device()==q.device()&&r.scalar_type()==torch::kFloat32&&r.is_contiguous()&&r.dim()==2&&r.size(0)==n&&r.size(1)==n,"Invalid Gram factor");
 int lines=q.numel()/n;decode_axis<<<(lines+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<float>(),r.data_ptr<float>(),q.size(0),q.size(1),q.size(2),axis);}
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("sample",&sample);m.def("decode",&decode);}
