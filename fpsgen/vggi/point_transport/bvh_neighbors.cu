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
  int batch=blockIdx.y;p+=int64_t(batch)*n*3;lo+=batch*3;hi+=batch*3;codes+=int64_t(batch)*n;
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
  int batch=blockIdx.y;p+=int64_t(batch)*n*3;ids+=int64_t(batch)*n;boxes+=int64_t(batch)*2*leaves*6;minima+=int64_t(batch)*2*leaves;
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

__global__ void parents_kernel(float* boxes,int64_t* minima,int first,int count,int stride) {
  boxes+=int64_t(blockIdx.y)*stride*6;minima+=int64_t(blockIdx.y)*stride;
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
  for(int count=leaves/2;count;count/=2)parents_kernel<<<(count+255)/256,256,0,stream>>>(boxes.data_ptr<float>(),minima.data_ptr<int64_t>(),count,count,0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {boxes,minima};
}


// A max heap costs logarithmic work per admitted candidate, rather than K moves.
__device__ bool worse(float a,int ai,float b,int bi) {
  return a>b || (a==b && ai>bi);
}
template<int K> __device__ void offer(float d,int id,float* ds,int* ids) {
  if(!worse(ds[0],ids[0],d,id))return;
  int pos=0;
  while(2*pos+1<K) {
    int child=2*pos+1;
    if(child+1<K && worse(ds[child+1],ids[child+1],ds[child],ids[child]))++child;
    if(!worse(ds[child],ids[child],d,id))break;
    ds[pos]=ds[child];ids[pos]=ids[child];pos=child;
  }
  ds[pos]=d;ids[pos]=id;
}
template<int K> __global__ void knn_kernel(
 const float* qp,const int64_t* qids,int nq,const float* sp,const int64_t* sids,int ns,
 const float* boxes,const int64_t* minima,int leaves,float* out,int64_t* nearest) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nq)return;
 float q[3]={qp[3*i],qp[3*i+1],qp[3*i+2]};
 float ds[K];int ids[K];
 #pragma unroll
 for(int k=0;k<K;k++){ds[k]=INFINITY;ids[k]=INT_MAX;}
 int stack[32],size=1;stack[0]=1;
 while(size) {
   int node=stack[--size];
   if(ds[0]==0.f && minima[node]>=ids[0])continue;
   float slack=8.f*FLT_EPSILON*ds[0]+1e-30f;
   if(lower_bound(boxes+6*node,q)>ds[0]+slack)continue;
   if(node>=leaves) {
     int leaf=node-leaves,begin=int64_t(leaf)*ns/leaves,end=int64_t(leaf+1)*ns/leaves;
     for(int j=begin;j<end;j++)offer<K>(distance2(q[0],q[1],q[2],sp+3*j),(int)sids[j],ds,ids);
   } else {
     int left=2*node,right=left+1;
     float dl=lower_bound(boxes+6*left,q),dr=lower_bound(boxes+6*right,q);
     if(dl<dr || (dl==dr && minima[left]<minima[right])) {
       if(dr<=ds[0]+slack)stack[size++]=right;
       if(dl<=ds[0]+slack)stack[size++]=left;
     } else {
       if(dl<=ds[0]+slack)stack[size++]=left;
       if(dr<=ds[0]+slack)stack[size++]=right;
     }
   }
 }
 // Heap sort only once at output, ascending distance then original ID.
 for(int end=K-1;end>0;end--) {
   float best=ds[0],d=ds[end];int bid=ids[0],id=ids[end];int pos=0;
   while(2*pos+1<end) {
     int child=2*pos+1;
     if(child+1<end && worse(ds[child+1],ids[child+1],ds[child],ids[child]))++child;
     if(!worse(ds[child],ids[child],d,id))break;
     ds[pos]=ds[child];ids[pos]=ids[child];pos=child;
   }
   ds[pos]=d;ids[pos]=id;ds[end]=best;ids[end]=bid;
 }
 int64_t off=qids[i]*K;
 for(int k=0;k<K;k++){out[off+k]=ds[k];nearest[off+k]=ids[k];}
}

__device__ void swap_lane(float &d,int &id,int mask,bool keep_min) {
 float other=__shfl_xor_sync(0xffffffff,d,mask);int oi=__shfl_xor_sync(0xffffffff,id,mask);
 bool own_worse=worse(d,id,other,oi);
 if(own_worse==keep_min){d=other;id=oi;}
}
__device__ void sort32(float &d,int &id,int lane) {
 #pragma unroll
 for(int size=2;size<=32;size*=2) {
   #pragma unroll
   for(int stride=size/2;stride;stride/=2)swap_lane(d,id,stride,((lane&stride)==0)==((lane&size)==0));
 }
}
__device__ void merge32(float &d,int &id,int lane) {
 #pragma unroll
 for(int stride=16;stride;stride/=2)swap_lane(d,id,stride,(lane&stride)==0);
}
template<int K> __global__ void warp_knn_kernel(
 const float* qp,const int64_t* qids,int nq,const float* sp,const int64_t* sids,int ns,
 const float* boxes,const int64_t* minima,int leaves,float* out,int64_t* nearest,const float* bounds=nullptr) {
 int batch=blockIdx.y;qp+=int64_t(batch)*nq*3;qids+=int64_t(batch)*nq;sp+=int64_t(batch)*ns*3;sids+=int64_t(batch)*ns;boxes+=int64_t(batch)*2*leaves*6;minima+=int64_t(batch)*2*leaves;out+=int64_t(batch)*nq*K;nearest+=int64_t(batch)*nq*K;
 int thread=blockIdx.x*blockDim.x+threadIdx.x,i=thread/32,lane=thread%32;if(i>=nq)return;
 float q[3]={qp[3*i],qp[3*i+1],qp[3*i+2]};
 float cap=bounds?bounds[int64_t(batch)*nq+qids[i]]:INFINITY;
 float a=INFINITY,b=INFINITY;int ai=INT_MAX,bi=INT_MAX;
 int stack[32],size=0;
 if(lane==0){stack[0]=1;size=1;}
 while(__shfl_sync(0xffffffff,size,0)) {
   int node=0;if(lane==0)node=stack[--size];node=__shfl_sync(0xffffffff,node,0);
   float worst=__shfl_sync(0xffffffff,K==32?a:b,31);worst=fminf(worst,cap);int worstid=__shfl_sync(0xffffffff,K==32?ai:bi,31);
   int visit=0;
   if(lane==0){float slack=8.f*FLT_EPSILON*worst+1e-30f;visit=!(worst==0 && minima[node]>=worstid) && lower_bound(boxes+6*node,q)<=worst+slack;}
   if(!__shfl_sync(0xffffffff,visit,0))continue;
   if(node>=leaves) {
     int leaf=node-leaves,begin=int64_t(leaf)*ns/leaves,end=int64_t(leaf+1)*ns/leaves;
     for(int base=begin;base<end;base+=32) {
       int j=base+lane;float d=INFINITY;int id=INT_MAX;
       if(j<end){d=distance2(q[0],q[1],q[2],sp+3*j);id=sids[j];}
       if(!__any_sync(0xffffffff,worse(worst,worstid,d,id)))continue;
       sort32(d,id,lane);
       float rd=__shfl_sync(0xffffffff,d,31-lane);int rid=__shfl_sync(0xffffffff,id,31-lane);
       if(K==32) {
         if(worse(a,ai,rd,rid)){a=rd;ai=rid;}
         merge32(a,ai,lane);
       } else {
         if(worse(b,bi,rd,rid)){b=rd;bi=rid;}
         if(worse(a,ai,b,bi)){float t=a;a=b;b=t;int ti=ai;ai=bi;bi=ti;}
         merge32(a,ai,lane);merge32(b,bi,lane);
       }
       worst=__shfl_sync(0xffffffff,K==32?a:b,31);worst=fminf(worst,cap);worstid=__shfl_sync(0xffffffff,K==32?ai:bi,31);
     }
   } else if(lane==0) {
     int left=2*node,right=left+1;
     float slack=8.f*FLT_EPSILON*worst+1e-30f;
     float dl=lower_bound(boxes+6*left,q),dr=lower_bound(boxes+6*right,q);
     if(dl<dr || (dl==dr && minima[left]<minima[right])) {
       if(dr<=worst+slack)stack[size++]=right;
       if(dl<=worst+slack)stack[size++]=left;
     } else {
       if(dl<=worst+slack)stack[size++]=left;
       if(dr<=worst+slack)stack[size++]=right;
     }
   }
 }
 int64_t off=qids[i]*K+lane;out[off]=a;nearest[off]=ai;
 if(K==64){out[off+32]=b;nearest[off+32]=bi;}
}
std::vector<torch::Tensor> knn(torch::Tensor qp,torch::Tensor qids,torch::Tensor sp,torch::Tensor sids,
 torch::Tensor boxes,torch::Tensor minima,int64_t leaves,int64_t k) {
 check_xyz(qp);check_xyz(sp);
 TORCH_CHECK(qp.device()==sp.device() && k<=sp.size(0),"Device/count mismatch");
 TORCH_CHECK(qids.device()==qp.device() && sids.device()==qp.device() && boxes.device()==qp.device() && minima.device()==qp.device(),"Index device mismatch");
 TORCH_CHECK(qids.scalar_type()==torch::kInt64 && sids.scalar_type()==torch::kInt64 && minima.scalar_type()==torch::kInt64 && boxes.scalar_type()==torch::kFloat32 && qids.is_contiguous() && sids.is_contiguous() && minima.is_contiguous() && boxes.is_contiguous(),"Invalid index layout");
 TORCH_CHECK(qids.numel()==qp.size(0) && sids.numel()==sp.size(0) && leaves>0 && leaves<=sp.size(0) && (leaves&(leaves-1))==0 && leaves<=(1<<28) && boxes.dim()==2 && boxes.size(0)==2*leaves && boxes.size(1)==6 && minima.numel()==2*leaves,"Invalid index shape");
 c10::cuda::CUDAGuard guard(qp.device());auto stream=c10::cuda::getCurrentCUDAStream(qp.get_device());
 auto d=torch::empty({qp.size(0),k},qp.options());auto ids=torch::empty({qp.size(0),k},qids.options());
 #define RUN(K) knn_kernel<K><<<(qp.size(0)+127)/128,128,0,stream>>>(qp.data_ptr<float>(),qids.data_ptr<int64_t>(),qp.size(0),sp.data_ptr<float>(),sids.data_ptr<int64_t>(),sp.size(0),boxes.data_ptr<float>(),minima.data_ptr<int64_t>(),leaves,d.data_ptr<float>(),ids.data_ptr<int64_t>())
 #define WARP(K) warp_knn_kernel<K><<<(qp.size(0)+3)/4,128,0,stream>>>(qp.data_ptr<float>(),qids.data_ptr<int64_t>(),qp.size(0),sp.data_ptr<float>(),sids.data_ptr<int64_t>(),sp.size(0),boxes.data_ptr<float>(),minima.data_ptr<int64_t>(),leaves,d.data_ptr<float>(),ids.data_ptr<int64_t>())
 if(k==1){RUN(1);}else if(k==32){WARP(32);}else if(k==64){WARP(64);}else TORCH_CHECK(false,"K=1/32/64 required");
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {d,ids};
}

// Equal-size scenes: grid.y selects an independent tree, never spatial offsets.
void check_batch_xyz(torch::Tensor p){
 TORCH_CHECK(p.is_cuda() && p.scalar_type()==torch::kFloat32 && p.is_contiguous()
 && p.dim()==3 && p.size(2)==3 && p.size(0)>0 && p.size(0)<=65535 && p.size(1)>0
 && p.size(0)*p.size(1)<INT_MAX,"Contiguous CUDA FP32 [B,N,3] required");
}
torch::Tensor batch_morton(torch::Tensor p,torch::Tensor lo,torch::Tensor hi){
 check_batch_xyz(p);int b=p.size(0),n=p.size(1);
 TORCH_CHECK(lo.device()==p.device() && hi.device()==p.device() && lo.scalar_type()==torch::kFloat32 && hi.scalar_type()==torch::kFloat32 && lo.is_contiguous() && hi.is_contiguous() && lo.numel()==b*3 && hi.numel()==b*3,"Invalid batched bounds");
 c10::cuda::CUDAGuard guard(p.device());auto stream=c10::cuda::getCurrentCUDAStream();
 auto codes=torch::empty({b,n},p.options().dtype(torch::kInt64));
 morton_kernel<<<dim3((n+255)/256,b),256,0,stream>>>(p.data_ptr<float>(),n,lo.data_ptr<float>(),hi.data_ptr<float>(),codes.data_ptr<int64_t>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return codes;
}
std::vector<torch::Tensor> batch_build(torch::Tensor p,torch::Tensor ids,int64_t leaves){
 check_batch_xyz(p);int b=p.size(0),n=p.size(1);
 TORCH_CHECK(ids.device()==p.device() && ids.scalar_type()==torch::kInt64 && ids.is_contiguous() && ids.numel()==int64_t(b)*n && leaves>0 && leaves<=n && (leaves&(leaves-1))==0 && leaves<=(1<<28),"Invalid batched tree");
 c10::cuda::CUDAGuard guard(p.device());auto stream=c10::cuda::getCurrentCUDAStream();
 auto boxes=torch::empty({b,2*leaves,6},p.options()),minima=torch::empty({b,2*leaves},ids.options());
 leaves_kernel<<<dim3((leaves+255)/256,b),256,0,stream>>>(p.data_ptr<float>(),ids.data_ptr<int64_t>(),n,leaves,boxes.data_ptr<float>(),minima.data_ptr<int64_t>());
 for(int count=leaves/2;count;count/=2)parents_kernel<<<dim3((count+255)/256,b),256,0,stream>>>(boxes.data_ptr<float>(),minima.data_ptr<int64_t>(),count,count,2*leaves);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {boxes,minima};
}
std::vector<torch::Tensor> batch_knn_bound(torch::Tensor qp,torch::Tensor qids,torch::Tensor sp,torch::Tensor sids,torch::Tensor boxes,torch::Tensor minima,int64_t leaves,int64_t k,torch::Tensor bounds) {
 check_batch_xyz(qp);check_batch_xyz(sp);int b=qp.size(0),nq=qp.size(1),ns=sp.size(1);
 TORCH_CHECK(qp.device()==sp.device() && sp.size(0)==b && (k==32||k==64) && k<=ns,"Batch query mismatch");
 TORCH_CHECK(qids.device()==qp.device() && sids.device()==qp.device() && boxes.device()==qp.device() && minima.device()==qp.device() && qids.scalar_type()==torch::kInt64 && sids.scalar_type()==torch::kInt64 && minima.scalar_type()==torch::kInt64 && boxes.scalar_type()==torch::kFloat32 && qids.is_contiguous() && sids.is_contiguous() && boxes.is_contiguous() && minima.is_contiguous(),"Invalid batched index tensors");
 TORCH_CHECK(qids.numel()==int64_t(b)*nq && sids.numel()==int64_t(b)*ns && leaves>0 && leaves<=ns && (leaves&(leaves-1))==0 && leaves<=(1<<28) && boxes.numel()==int64_t(b)*2*leaves*6 && minima.numel()==int64_t(b)*2*leaves,"Invalid batched index sizes");
 c10::cuda::CUDAGuard guard(qp.device());auto stream=c10::cuda::getCurrentCUDAStream();
 if(bounds.defined())TORCH_CHECK(bounds.device()==qp.device()&&bounds.scalar_type()==torch::kFloat32&&bounds.is_contiguous()&&bounds.numel()==int64_t(b)*nq,"Invalid search bounds");
 auto d=torch::empty({b,nq,k},qp.options()),ids=torch::empty({b,nq,k},qids.options());
 #define BATCH_WARP(K) warp_knn_kernel<K><<<dim3((nq+3)/4,b),128,0,stream>>>(qp.data_ptr<float>(),qids.data_ptr<int64_t>(),nq,sp.data_ptr<float>(),sids.data_ptr<int64_t>(),ns,boxes.data_ptr<float>(),minima.data_ptr<int64_t>(),leaves,d.data_ptr<float>(),ids.data_ptr<int64_t>(),bounds.defined()?bounds.data_ptr<float>():nullptr)
 if(k==32){BATCH_WARP(32);}else{BATCH_WARP(64);}
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {d,ids};
}

std::vector<torch::Tensor> batch_knn(torch::Tensor qp,torch::Tensor qids,torch::Tensor sp,torch::Tensor sids,torch::Tensor boxes,torch::Tensor minima,int64_t leaves,int64_t k){return batch_knn_bound(qp,qids,sp,sids,boxes,minima,leaves,k,torch::Tensor());}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("morton",&morton);m.def("build_bvh",&build_bvh);m.def("knn",&knn);m.def("batch_morton",&batch_morton);m.def("batch_build",&batch_build);m.def("batch_knn",&batch_knn);m.def("batch_knn_bound",&batch_knn_bound);}
