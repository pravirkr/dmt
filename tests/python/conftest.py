import sys
from pathlib import Path

# Ensure the local build/src and src directories take precedence over stale editable site-packages
repo_root = Path(__file__).resolve().parents[2]
build_src = repo_root / "build" / "src"
src_dir = repo_root / "src"

sys.meta_path = [
    f for f in sys.meta_path if "ScikitBuildRedirectingFinder" not in type(f).__name__
]

if str(build_src) not in sys.path:
    sys.path.insert(0, str(build_src))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
