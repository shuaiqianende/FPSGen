from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import yaml
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry


def geometry(path):
    return SparseHierarchyGeometry(**yaml.safe_load(Path(path).read_text()))
