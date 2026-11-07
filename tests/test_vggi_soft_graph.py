import pytest
import torch
from fpsgen.vggi.point_transport.bvh_neighbors import BVHIndex
from fpsgen.vggi.point_transport.dynamic_soft import graph,solve_graph
from fpsgen.vggi.point_transport.soft_flow import match

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_decode_large_64bit_keys():
    from fpsgen.vggi.point_transport.soft_graph import decode
    torch.manual_seed(513);n=180000
    x=torch.randn(n,3,device='cuda');y=torch.randn_like(x)
    keys=torch.tensor([0,n-1,(n-1)*n,n*n-1,54321*n+98765],device='cuda',dtype=torch.int64)
    s,t,c=decode(keys,x,y)
    assert torch.equal(s,keys//n) and torch.equal(t,keys%n)
    torch.testing.assert_close(c,.5*(x[s]-y[t]).square().sum(1),rtol=0,atol=0)


@pytest.mark.parametrize('self_graph,k',[(False,32),(False,64),(True,32),(True,64)])
def test_exact_edges_costs_and_transpose(self_graph,k):
    torch.manual_seed(432);x=torch.randn(137,3,device='cuda');x[10:15]=x[0]
    y=x if self_graph else torch.randn_like(x)
    a=graph(x,y,x,k)
    xi,yi=BVHIndex(x),None
    yi=xi if self_graph else BVHIndex(y)
    b=graph(x,y,x,k,index_y=yi,index_query=xi,layout='scatter')
    for field in ('keys','offsets','offsets_t','cost'):torch.testing.assert_close(a[field],b[field],rtol=0,atol=0)
    cols=torch.arange(len(x),device=x.device).repeat_interleave(b['offsets_t'][1:]-b['offsets_t'][:-1])
    order=(b['sc']*len(x)+cols).argsort()
    torch.testing.assert_close((b['sc']*len(x)+cols)[order],b['keys'],rtol=0,atol=0)
    torch.testing.assert_close(b['cost_t'][order],b['cost'],rtol=0,atol=0)
    za,_,_=solve_graph(x,y,a,epsilon=.02,penalty=50,iterations=30)
    zb,_,_=solve_graph(x,y,b,epsilon=.02,penalty=50,iterations=30)
    torch.testing.assert_close(za,zb,rtol=3e-4,atol=3e-5)


def test_full_soft_flow_backend_equivalence():
    torch.manual_seed(28);x=torch.randn(139,3,device='cuda');y=torch.randn_like(x)
    out={}
    for backend in ('legacy','shared','fast'):
        with torch.cuda.amp.autocast():
            out[backend],info=match(x,y,k=32,epsilons=(.02,.005,.002),penalty=50,
                                    iterations=30,graph_backend=backend)
        assert not info['hard_assignment'] and out[backend].dtype==torch.float32
    torch.testing.assert_close(out['legacy'],out['shared'],rtol=1e-4,atol=3e-5)
    torch.testing.assert_close(out['legacy'],out['fast'],rtol=5e-4,atol=1e-4)
