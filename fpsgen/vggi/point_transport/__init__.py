"""Recommended continuous soft transport; no hard assignments."""
def match(source,target,**kwargs):
    from .soft_flow import match as impl
    return impl(source,target,**kwargs)
