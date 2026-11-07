#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Sorted grid index, expanding Chebyshev shells, exact distance certificate.
// Rare uncertified queries fall back to all reference points; never truncate K.
template<int K> __device__ void insert(float d, int id, float* ds, int* ids) {
  if(d>=ds[K-1])return;
  int p=K-1;
  while(p>0 && d<ds[p-1]){ds[p]=ds[p-1];ids[p]=ids[p-1];--p;}
  ds[p]=d;ids[p]=id;
}
template<int K> __global__ void search_kernel(
 const float* q,const float* ref,const int64_t* order,const int64_t* offsets,
 const float* origin,int nx,int ny,int nz,float cell,int n,int m,int max_shell,
 int64_t* out,float* dists,int* shells,int* fallback) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  float x=q[3*i],y=q[3*i+1],z=q[3*i+2];
  int cx=floorf((x-origin[0])/cell),cy=floorf((y-origin[1])/cell),cz=floorf((z-origin[2])/cell);
  float frac[3]={x-origin[0]-cx*cell,y-origin[1]-cy*cell,z-origin[2]-cz*cell};
  float ds[K];int ids[K];for(int k=0;k<K;k++){ds[k]=INFINITY;ids[k]=-1;}
  bool done=false;int r=0;
  for(r=0;r<=max_shell;r++) {
    int za=max(0,cz-r),zb=min(nz-1,cz+r);
    int ya=max(0,cy-r),yb=min(ny-1,cy+r);
    int xa=max(0,cx-r),xb=min(nx-1,cx+r);
    for(int iz=za;iz<=zb;iz++)for(int iy=ya;iy<=yb;iy++)for(int ix=xa;ix<=xb;ix++) {
      if(r>0 && max(max(abs(ix-cx),abs(iy-cy)),abs(iz-cz))!=r)continue;
      int key=(iz*ny+iy)*nx+ix;
      for(int64_t j=offsets[key];j<offsets[key+1];j++) {
        float dx=x-ref[3*j],dy=y-ref[3*j+1],dz=z-ref[3*j+2];
        insert<K>(dx*dx+dy*dy+dz*dz,(int)order[j],ds,ids);
      }
    }
    float bound=INFINITY;
    for(int a=0;a<3;a++)bound=fminf(bound,fminf(r*cell+frac[a],(r+1)*cell-frac[a]));
    bool whole=(cx-r<=0 && cy-r<=0 && cz-r<=0 && cx+r>=nx-1 && cy+r>=ny-1 && cz+r>=nz-1);
    if(ids[K-1]>=0 && (ds[K-1]<=bound*bound || whole)){done=true;break;}
  }
  if(!done) {
    // Reset to avoid counting already visited references twice.
    for(int k=0;k<K;k++){ds[k]=INFINITY;ids[k]=-1;}
    for(int j=0;j<m;j++) {
      float dx=x-ref[3*j],dy=y-ref[3*j+1],dz=z-ref[3*j+2];
      insert<K>(dx*dx+dy*dy+dz*dz,(int)order[j],ds,ids);
    }
  }
  shells[i]=r;fallback[i]=!done;
  for(int k=0;k<K;k++){out[i*K+k]=ids[k];dists[i*K+k]=ds[k];}
}

void knn(torch::Tensor q,torch::Tensor ref,torch::Tensor order,torch::Tensor offsets,
 torch::Tensor origin,int nx,int ny,int nz,double cell,int max_shell,
 torch::Tensor out,torch::Tensor distances,torch::Tensor shells,torch::Tensor fallback) {
  c10::cuda::CUDAGuard guard(q.device());
  int n=q.size(0),m=ref.size(0),k=out.size(1);auto stream=at::cuda::getCurrentCUDAStream();
  #define RUN(K) search_kernel<K><<<(n+127)/128,128,0,stream>>>(q.data_ptr<float>(),ref.data_ptr<float>(),order.data_ptr<int64_t>(),offsets.data_ptr<int64_t>(),origin.data_ptr<float>(),nx,ny,nz,(float)cell,n,m,max_shell,out.data_ptr<int64_t>(),distances.data_ptr<float>(),shells.data_ptr<int>(),fallback.data_ptr<int>())
  if(k==1){RUN(1);}else if(k==32){RUN(32);}else if(k==64){RUN(64);}else TORCH_CHECK(false,"Unsupported K");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void dual_kernel(const int64_t* offsets,const int64_t* neighbors,const float* cost,
 const float* other,float* dual,int n,float eps) {
  // One warp per row handles the heavy-tailed degrees of reverse kNN edges.
  int tid=blockIdx.x*blockDim.x+threadIdx.x,i=tid/32,lane=tid%32;if(i>=n)return;
  int64_t a=offsets[i],b=offsets[i+1];float mx=-INFINITY;
  for(int64_t e=a+lane;e<b;e+=32)mx=fmaxf(mx,(other[neighbors[e]]-cost[e])/eps);
  for(int d=16;d>0;d/=2)mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,d));
  mx=__shfl_sync(0xffffffff,mx,0);
  float sum=0;for(int64_t e=a+lane;e<b;e+=32)sum+=expf((other[neighbors[e]]-cost[e])/eps-mx);
  for(int d=16;d>0;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);
  if(lane==0)dual[i]=-eps*(mx+logf(sum));
}
void dual(torch::Tensor offsets,torch::Tensor neighbors,torch::Tensor cost,
 torch::Tensor other,torch::Tensor result,double eps) {
  c10::cuda::CUDAGuard guard(result.device());int n=result.numel();
  dual_kernel<<<(n+3)/4,128,0,at::cuda::getCurrentCUDAStream()>>>(offsets.data_ptr<int64_t>(),neighbors.data_ptr<int64_t>(),cost.data_ptr<float>(),other.data_ptr<float>(),result.data_ptr<float>(),n,eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("knn",&knn);m.def("dual",&dual);}
