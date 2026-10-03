"""Partition BEV parameters without double counting the shared frontend."""


def parameter_counts(model):
    count = lambda module: sum(p.numel() for p in module.parameters())
    total = count(model)
    pillar = count(model.pc_encoder)
    if hasattr(model, "core"):
        core = count(model.core)
        condition = total - pillar - core
    else:
        # Legacy owns the dense modules directly. Condition pyramid and
        # stage fusion projections are adapters; transformer cross-attention
        # remains part of the core (reported explicitly in the study docs).
        adapters = [model.cond_fpn, model.mid_fuse]
        adapters += [stage["cond_fuse"] for stage in (*model.downs, *model.ups)]
        condition = sum(count(module) for module in adapters)
        core = total - pillar - condition
    assert min(core, condition, pillar) >= 0
    assert core + condition + pillar == total
    return {"core": core, "condition": condition, "pointpillar": pillar,
            "total": total, "total_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def source_fingerprint(root):
    """Hash executable local Python sources, including dirty/untracked code."""
    import hashlib
    from pathlib import Path
    root = Path(root)
    paths = sorted((root / 'fpsgen').rglob('*.py'))
    paths += sorted((root / 'scripts').glob('*.py'))
    entries = {}
    for path in paths:
        if path.is_file():
            entries[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    aggregate = hashlib.sha256()
    for name, digest in sorted(entries.items()):
        aggregate.update(f'{name}\0{digest}\n'.encode())
    return {'sha256': aggregate.hexdigest(), 'files': entries}
