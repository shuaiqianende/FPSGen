// Independent scene-scale Chamfer kernels. Original chamfer_3D sources unchanged.
// Squared nearest-distance and atomic coordinate derivatives follow the standard
// Chamfer formula (see ChamferDistancePytorch, MIT, Thibault GROUEIX, 2019).
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <vector>

void check_xyz(torch::Tensor p) {
  TORCH_CHECK(p.is_cuda() && p.scalar_type()==torch::kFloat32 && p.is_contiguous(),"Contiguous CUDA FP32 required");
  TORCH_CHECK(p.dim()==2 && p.size(1)==3 && p.size(0)>0 && p.size(0)<INT_MAX,"Nonempty [N,3], N<INT_MAX required");
}

__device__ unsigned long long spread21(unsigned int v) {
  unsigned long long x=v&0x1fffff;
  x=(x|(x<<32))&0x1f00000000ffffULL;
  x=(x|(x<<16))&0x1f0000ff0000ffULL;
  x=(x|(x<<8))&0x100f00f00f00f00fULL;
  x=(x|(x<<4))&0x10c30c30c30c30c3ULL;
  return (x|(x<<2))&0x1249249249249249ULL;
}

__global__ void morton_kernel(const float* p,int n,const float* lo,const float* hi,int64_t* codes) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  unsigned int v[3];
  for(int a=0;a<3;a++) {
    float length=fmaxf(hi[a]-lo[a],1e-20f);
    float x=fminf(fmaxf((p[3*i+a]-lo[a])/length,0.f),1.f);
    v[a]=(unsigned int)(x*2097151.f);
  }
  codes[i]=(int64_t)(spread21(v[0])|(spread21(v[1])<<1)|(spread21(v[2])<<2));
}

__global__ void leaves_kernel(const float* p,const int64_t* ids,int n,int leaves,float* boxes,int64_t* minima) {
  int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=leaves)return;
  int begin=int64_t(j)*n/leaves,end=int64_t(j+1)*n/leaves;
  float lo[3]={INFINITY,INFINITY,INFINITY},hi[3]={-INFINITY,-INFINITY,-INFINITY};
  int64_t minimum=INT64_MAX;
  for(int i=begin;i<end;i++) {
    minimum=min(minimum,ids[i]);
    for(int a=0;a<3;a++){lo[a]=fminf(lo[a],p[3*i+a]);hi[a]=fmaxf(hi[a],p[3*i+a]);}
  }
  int node=leaves+j;minima[node]=minimum;
  for(int a=0;a<3;a++){boxes[6*node+a]=lo[a];boxes[6*node+3+a]=hi[a];}
}

__global__ void parents_kernel(float* boxes,int64_t* minima,int first,int count) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
  int node=first+i;minima[node]=min(minima[node*2],minima[node*2+1]);
  for(int a=0;a<3;a++) {
    boxes[6*node+a]=fminf(boxes[12*node+a],boxes[12*node+6+a]);
    boxes[6*node+3+a]=fmaxf(boxes[12*node+3+a],boxes[12*node+9+a]);
  }
}

__device__ float lower_bound(const float* box,const float* q) {
  float result=0.f;
  for(int a=0;a<3;a++) {
    float d=fmaxf(0.f,fmaxf(__fsub_rd(box[a],q[a]),__fsub_rd(q[a],box[3+a])));
    result=__fadd_rd(result,__fmul_rd(d,d));
  }
  return result;
}

__device__ float distance2(float x,float y,float z,const float* p) {
  float dx=x-p[0],dy=y-p[1],dz=z-p[2];
  return (dx*dx+dy*dy)+dz*dz;
}

__global__ void bvh_query_kernel(const float* qp,const int64_t* qids,int nq,
    const float* sp,const int64_t* sids,int ns,const float* boxes,const int64_t* minima,
    int leaves,float* out,int64_t* nearest) {
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<nq;i+=blockDim.x*gridDim.x) {
    float q[3]={qp[3*i],qp[3*i+1],qp[3*i+2]};
    float best=INFINITY;int64_t best_id=INT64_MAX;
    int stack[32],size=1;stack[0]=1;
    while(size) {
      int node=stack[--size];
      // Preserve exact-zero minimum-index ties without scanning duplicate clouds.
      if(best==0.f && minima[node]>=best_id)continue;
      float slack=8.f*FLT_EPSILON*best+1e-30f;
      if(lower_bound(boxes+6*node,q)>best+slack)continue;
      if(node>=leaves) {
        int leaf=node-leaves,begin=int64_t(leaf)*ns/leaves,end=int64_t(leaf+1)*ns/leaves;
        for(int j=begin;j<end;j++) {
          float d=distance2(q[0],q[1],q[2],sp+3*j);
          if(d<best || (d==best && sids[j]<best_id)){best=d;best_id=sids[j];}
        }
      } else {
        int left=2*node,right=left+1;
        float dl=lower_bound(boxes+6*left,q),dr=lower_bound(boxes+6*right,q);
        if(dl<dr || (dl==dr && minima[left]<minima[right])) {
          if(dr<=best+slack)stack[size++]=right;
          if(dl<=best+slack)stack[size++]=left;
        } else {
          if(dl<=best+slack)stack[size++]=left;
          if(dr<=best+slack)stack[size++]=right;
        }
      }
    }
    out[qids[i]]=best;nearest[qids[i]]=best_id;
  }
}

// Dynamic query grid: thousands of active blocks for scene B1, unlike fixed16.
// Each query's minimum stays in registers over every support tile.
template<int TILE>
__global__ void tiled_query_kernel(const float* qp,int nq,const float* sp,int ns,float* out,int64_t* nearest) {
  __shared__ float sx[TILE],sy[TILE],sz[TILE];
  int lane=threadIdx.x,i=blockIdx.x*TILE+lane;
  // Retain a valid original index even when finite coordinates overflow FP32 d2.
  float x=0,y=0,z=0,best=INFINITY;int64_t best_id=0;
  if(i<nq){x=qp[3*i];y=qp[3*i+1];z=qp[3*i+2];}
  for(int begin=0;begin<ns;begin+=TILE) {
    int j=begin+lane;if(j<ns){sx[lane]=sp[3*j];sy[lane]=sp[3*j+1];sz[lane]=sp[3*j+2];}
    __syncthreads();
    if(i<nq) {
      int end=min(TILE,ns-begin);
      #pragma unroll 4
      for(int k=0;k<end;k++) {
        float dx=x-sx[k],dy=y-sy[k],dz=z-sz[k];float d=(dx*dx+dy*dy)+dz*dz;
        if(d<best){best=d;best_id=begin+k;}
      }
    }
    __syncthreads();
  }
  if(i<nq){out[i]=best;nearest[i]=best_id;}
}

__global__ void backward_direction(const float* q,const float* s,const int64_t* nearest,
    const float* upstream,int n,float* gq,float* gs) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  int64_t j=nearest[i];float weight=2.f*upstream[i];
  for(int a=0;a<3;a++) {
    float value=weight*(q[3*i+a]-s[3*j+a]);
    atomicAdd(gq+3*i+a,value);atomicAdd(gs+3*j+a,-value);
  }
}

torch::Tensor morton(torch::Tensor p,torch::Tensor lo,torch::Tensor hi) {
  check_xyz(p);TORCH_CHECK(lo.device()==p.device() && hi.device()==p.device() && lo.numel()==3 && hi.numel()==3 && lo.scalar_type()==torch::kFloat32 && hi.scalar_type()==torch::kFloat32 && lo.is_contiguous() && hi.is_contiguous(),"Invalid bounds");
  c10::cuda::CUDAGuard guard(p.device());auto stream=c10::cuda::getCurrentCUDAStream(p.get_device());
  auto codes=torch::empty({p.size(0)},p.options().dtype(torch::kInt64));
  morton_kernel<<<(p.size(0)+255)/256,256,0,stream>>>(p.data_ptr<float>(),p.size(0),lo.data_ptr<float>(),hi.data_ptr<float>(),codes.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return codes;
}

std::vector<torch::Tensor> build_bvh(torch::Tensor p,torch::Tensor ids,int64_t leaves) {
  check_xyz(p);
  TORCH_CHECK(ids.device()==p.device() && ids.scalar_type()==torch::kInt64 && ids.is_contiguous() && ids.numel()==p.size(0),"Invalid IDs");
  TORCH_CHECK(leaves>0 && leaves<=p.size(0) && (leaves&(leaves-1))==0 && leaves<=(1<<28),"Invalid balanced leaves");
  c10::cuda::CUDAGuard guard(p.device());auto stream=c10::cuda::getCurrentCUDAStream(p.get_device());
  auto boxes=torch::empty({2*leaves,6},p.options());auto minima=torch::empty({2*leaves},ids.options());
  leaves_kernel<<<(leaves+255)/256,256,0,stream>>>(p.data_ptr<float>(),ids.data_ptr<int64_t>(),p.size(0),leaves,boxes.data_ptr<float>(),minima.data_ptr<int64_t>());
  for(int count=leaves/2;count;count/=2)parents_kernel<<<(count+255)/256,256,0,stream>>>(boxes.data_ptr<float>(),minima.data_ptr<int64_t>(),count,count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {boxes,minima};
}

std::vector<torch::Tensor> bvh_query(torch::Tensor qp,torch::Tensor qids,torch::Tensor sp,torch::Tensor sids,torch::Tensor boxes,torch::Tensor minima,int64_t leaves) {
  check_xyz(qp);check_xyz(sp);
  TORCH_CHECK(sp.device()==qp.device() && qids.device()==qp.device() && sids.device()==qp.device() && boxes.device()==qp.device() && minima.device()==qp.device(),"One device required");
  TORCH_CHECK(qids.is_contiguous() && sids.is_contiguous() && boxes.is_contiguous() && minima.is_contiguous() && qids.scalar_type()==torch::kInt64 && sids.scalar_type()==torch::kInt64 && minima.scalar_type()==torch::kInt64 && boxes.scalar_type()==torch::kFloat32,"Invalid index types");
  TORCH_CHECK(qids.numel()==qp.size(0) && sids.numel()==sp.size(0) && boxes.dim()==2 && boxes.size(0)==2*leaves && boxes.size(1)==6 && minima.numel()==2*leaves && leaves>0 && leaves<=(1<<28),"Invalid tree shapes");
  c10::cuda::CUDAGuard guard(qp.device());auto stream=c10::cuda::getCurrentCUDAStream(qp.get_device());
  auto d=torch::empty({qp.size(0)},qp.options());auto idx=torch::empty({qp.size(0)},qids.options());
  bvh_query_kernel<<<std::min<int64_t>((qp.size(0)+127)/128,65535),128,0,stream>>>(qp.data_ptr<float>(),qids.data_ptr<int64_t>(),qp.size(0),sp.data_ptr<float>(),sids.data_ptr<int64_t>(),sp.size(0),boxes.data_ptr<float>(),minima.data_ptr<int64_t>(),leaves,d.data_ptr<float>(),idx.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {d,idx};
}

std::vector<torch::Tensor> tiled(torch::Tensor p,torch::Tensor q) {
  check_xyz(p);check_xyz(q);TORCH_CHECK(p.device()==q.device(),"One device required");
  c10::cuda::CUDAGuard guard(p.device());auto stream=c10::cuda::getCurrentCUDAStream(p.get_device());
  auto d=torch::empty({p.size(0)},p.options());auto idx=torch::empty({p.size(0)},p.options().dtype(torch::kInt64));
  tiled_query_kernel<256><<<(p.size(0)+255)/256,256,0,stream>>>(p.data_ptr<float>(),p.size(0),q.data_ptr<float>(),q.size(0),d.data_ptr<float>(),idx.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {d,idx};
}

std::vector<torch::Tensor> backward(torch::Tensor p,torch::Tensor q,torch::Tensor ip,torch::Tensor iq,torch::Tensor dp,torch::Tensor dq) {
  check_xyz(p);check_xyz(q);TORCH_CHECK(p.device()==q.device() && ip.device()==p.device() && iq.device()==p.device() && dp.device()==p.device() && dq.device()==p.device(),"One device required");
  TORCH_CHECK(ip.scalar_type()==torch::kInt64 && iq.scalar_type()==torch::kInt64 && dp.scalar_type()==torch::kFloat32 && dq.scalar_type()==torch::kFloat32 && ip.is_contiguous() && iq.is_contiguous() && dp.is_contiguous() && dq.is_contiguous() && ip.numel()==p.size(0) && dp.numel()==p.size(0) && iq.numel()==q.size(0) && dq.numel()==q.size(0),"Invalid backward tensors");
  c10::cuda::CUDAGuard guard(p.device());auto stream=c10::cuda::getCurrentCUDAStream(p.get_device());
  auto gp=torch::zeros_like(p),gq=torch::zeros_like(q);
  backward_direction<<<(p.size(0)+255)/256,256,0,stream>>>(p.data_ptr<float>(),q.data_ptr<float>(),ip.data_ptr<int64_t>(),dp.data_ptr<float>(),p.size(0),gp.data_ptr<float>(),gq.data_ptr<float>());
  backward_direction<<<(q.size(0)+255)/256,256,0,stream>>>(q.data_ptr<float>(),p.data_ptr<float>(),iq.data_ptr<int64_t>(),dq.data_ptr<float>(),q.size(0),gq.data_ptr<float>(),gp.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {gp,gq};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("morton",&morton);m.def("build_bvh",&build_bvh);m.def("bvh_query",&bvh_query);m.def("tiled",&tiled);m.def("backward",&backward);
}
