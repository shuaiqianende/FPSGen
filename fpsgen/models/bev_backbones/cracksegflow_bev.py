"""Paper-based CrackSegFlow-style decoder-SPADE adaptation, not official code."""
from .synflow_bev import BEVSynFlowS


class BEVCrackSegFlowS(BEVSynFlowS):
    """Same SynFlow U-Net; config restricts SPADE to decoder stages."""
    pass
