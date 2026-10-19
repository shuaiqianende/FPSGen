#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

__global__ void leaf_boxes(const float* p, float* boxes, int64_t n, int leaves) {
  int leaf = blockIdx.x*blockDim.x+threadIdx.x;
  if (leaf >= leaves) return;
  int64_t start = int64_t(leaf)*n/leaves, end = int64_t(leaf+1)*n/leaves;
  float lo[3] = {INFINITY,INFINITY,INFINITY}, hi[3] = {-INFINITY,-INFINITY,-INFINITY};
  for (int64_t i=start;i<end;i++) for (int a=0;a<3;a++) {
    lo[a] = fminf(lo[a],p[3*i+a]); hi[a] = fmaxf(hi[a],p[3*i+a]);
  }
  for (int a=0;a<3;a++) { boxes[6*(leaves+leaf)+a]=lo[a]; boxes[6*(leaves+leaf)+3+a]=hi[a]; }
}

__global__ void parent_boxes(float* boxes, int first, int count) {
  int j=blockIdx.x*blockDim.x+threadIdx.x;
  if (j>=count) return;
  int parent=first+j;
  for (int a=0;a<3;a++) {
    boxes[6*parent+a]=fminf(boxes[12*parent+a],boxes[12*parent+6+a]);
    boxes[6*parent+3+a]=fmaxf(boxes[12*parent+3+a],boxes[12*parent+9+a]);
  }
}

__device__ double box_distance(const float* box, const double* query) {
  double lower=0.;
  for(int a=0;a<3;a++) {
    double d=fmax(double(box[a])-query[a],fmax(query[a]-double(box[3+a]),0.));
    // Downward rounding makes the box distance a conservative lower bound.
    lower=__dadd_rd(lower,__dmul_rd(d,d));
  }
  return lower;
}

__global__ void query_tree(const float* queries, int64_t nq, const float* points,
    int64_t np, const int64_t* original, const float* boxes, int leaves,
    double* distances, int64_t* ids) {
  for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<nq;i+=int64_t(gridDim.x)*blockDim.x) {
    double q[3]={queries[3*i],queries[3*i+1],queries[3*i+2]};
    double best=INFINITY; int64_t best_id=-1;
    int stack[32], size=1; stack[0]=1;
    while(size) {
      int node=stack[--size];
      // Small outward margin prevents round-off pruning a true nearest point.
      if(box_distance(boxes+6*node,q)>best*(1.+1e-14)+1e-30) continue;
      if(node>=leaves) {
        int leaf=node-leaves;
        int64_t start=int64_t(leaf)*np/leaves, end=int64_t(leaf+1)*np/leaves;
        for(int64_t j=start;j<end;j++) {
          double a=q[0]-double(points[3*j]), b=q[1]-double(points[3*j+1]), c=q[2]-double(points[3*j+2]);
          double d=a*a+b*b+c*c;
          if(d<best || (d==best && original[j]<best_id)) {best=d;best_id=original[j];}
        }
      } else {
        int left=2*node, right=left+1;
        double dl=box_distance(boxes+6*left,q), dr=box_distance(boxes+6*right,q);
        if(dl<dr) {if(dr<=best*(1.+1e-14)+1e-30) stack[size++]=right;
                   if(dl<=best*(1.+1e-14)+1e-30) stack[size++]=left;}
        else {if(dl<=best*(1.+1e-14)+1e-30) stack[size++]=left;
              if(dr<=best*(1.+1e-14)+1e-30) stack[size++]=right;}
      }
    }
    distances[i]=best;ids[i]=best_id;
  }
}

torch::Tensor build_boxes(torch::Tensor points, int64_t leaves) {
  TORCH_CHECK(points.is_cuda() && points.scalar_type()==torch::kFloat32 && points.is_contiguous(),"CUDA FP32 contiguous required");
  TORCH_CHECK(points.dim()==2 && points.size(1)==3 && points.size(0)>0,"Nonempty XYZ required");
  TORCH_CHECK(leaves>0 && leaves<=points.size(0) && (leaves&(leaves-1))==0 && leaves<=(1<<28),"Invalid leaves");
  c10::cuda::CUDAGuard guard(points.device());
  auto boxes=torch::empty({2*leaves,6},points.options());
  auto stream=c10::cuda::getCurrentCUDAStream(points.get_device());
  leaf_boxes<<<(leaves+255)/256,256,0,stream>>>(points.data_ptr<float>(),boxes.data_ptr<float>(),points.size(0),leaves);
  for(int count=leaves/2;count;count/=2)
    parent_boxes<<<(count+255)/256,256,0,stream>>>(boxes.data_ptr<float>(),count,count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return boxes;
}

std::vector<torch::Tensor> query(torch::Tensor queries,torch::Tensor points,
    torch::Tensor original,torch::Tensor boxes,int64_t leaves) {
  TORCH_CHECK(queries.is_cuda() && points.is_cuda() && original.is_cuda() && boxes.is_cuda(),"CUDA required");
  TORCH_CHECK(queries.device()==points.device() && original.device()==points.device() && boxes.device()==points.device(),"One device required");
  TORCH_CHECK(queries.scalar_type()==torch::kFloat32 && points.scalar_type()==torch::kFloat32 && original.scalar_type()==torch::kInt64 && boxes.scalar_type()==torch::kFloat32,"Invalid dtypes");
  TORCH_CHECK(queries.is_contiguous() && points.is_contiguous() && original.is_contiguous() && boxes.is_contiguous(),"Contiguous required");
  TORCH_CHECK(queries.dim()==2 && queries.size(1)==3 && points.dim()==2 && points.size(1)==3 && original.numel()==points.size(0) && boxes.size(0)==2*leaves && boxes.size(1)==6,"Invalid shapes");
  c10::cuda::CUDAGuard guard(points.device());
  auto distances=torch::empty({queries.size(0)},queries.options().dtype(torch::kFloat64));
  auto ids=torch::empty({queries.size(0)},queries.options().dtype(torch::kInt64));
  int blocks=std::min<int64_t>((queries.size(0)+127)/128,65535);
  if(blocks) query_tree<<<blocks,128,0,c10::cuda::getCurrentCUDAStream(points.get_device())>>>(queries.data_ptr<float>(),queries.size(0),points.data_ptr<float>(),points.size(0),original.data_ptr<int64_t>(),boxes.data_ptr<float>(),leaves,distances.data_ptr<double>(),ids.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {distances,ids};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("build_boxes",&build_boxes);m.def("query",&query);}
