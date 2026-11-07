import pytest
import torch
from fpsgen.vggi.point_transport.batch_soft import BatchBVH,graph as reference,solve
from fpsgen.vggi.point_transport.linear_graph import graph
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('k',[32,64])
@pytest.mark.parametrize('self_graph',[False,True])
def test_exact_union(k,self_graph):
    torch.manual_seed(7)
    x=torch.randn(2,137,3,device='cuda');y=x if self_graph else torch.randn_like(x)
    xi=BatchBVH(x);yi=xi if self_graph else BatchBVH(y)
    old=reference(x,y,xi,yi,k);new=graph(x,y,xi,yi,k);n=274
    for offsets,ids,costs,transpose in [('offsets','t','cost',False),('offsets_t','sc','cost_t',True)]:
        rows=torch.arange(n,device='cuda').repeat_interleave(new[offsets][1:]-new[offsets][:-1])
        cols=new[ids][:len(rows)].long()
        keys=cols*n+rows if transpose else rows*n+cols
        keys,order=keys.sort()
        assert torch.equal(keys,old['keys'])
        torch.testing.assert_close(new[costs][:len(rows)][order],old['cost'],rtol=0,atol=0)
    a=solve(x.reshape(-1,3),y.reshape(-1,3),old,.1,.3,50,2)
    b=solve(x.reshape(-1,3),y.reshape(-1,3),new,.1,.3,50,2)
    for u,v in zip(a,b):torch.testing.assert_close(u,v,rtol=2e-4,atol=3e-6)

@pytest.mark.parametrize('k',[32,64])
def test_previous_neighbors_are_only_exact_search_bounds(k):
    torch.manual_seed(44);x=torch.randn(2,137,3,device='cuda');y=torch.randn_like(x)
    old=BatchBVH(y).query(BatchBVH(x),k)
    x=x+torch.randn_like(x)*.7;y=y+torch.randn_like(y)*.7
    xi,yi=BatchBVH(x),BatchBVH(y)
    assert torch.equal(yi.query(xi,k),yi.query(xi,k,old))
    # A completely tied scene still respects the stable ID tie break.
    zero=torch.zeros_like(x);zi=BatchBVH(zero)
    assert torch.equal(zi.query(zi,k),zi.query(zi,k,old))
