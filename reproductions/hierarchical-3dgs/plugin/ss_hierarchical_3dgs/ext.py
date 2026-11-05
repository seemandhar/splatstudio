"""JIT builds of the vendored official CUDA/C++ code (run through tools/ssenv).

vendor/hierarchy_rasterizer  github.com/graphdeco-inria/hierarchy-rasterizer @63fa247, unmodified
                             (Inria 3DGS rasterizer + Mip-Splatting 2D filter + hierarchy
                             interpolation of alpha, Eq. alphap, with hand-written backward).
vendor/gaussian_hierarchy    github.com/graphdeco-inria/gaussian-hierarchy @677c855, unmodified
                             (hierarchy creation, cut selection expand_to_size, interpolation
                             weights get_interpolation_weights, .hier loader/writer).
vendor/third_party           glm (MIT, @6f14f47) and Eigen 3.4.0 (MPL2), header-only.
Licence of the Inria code: vendor/LICENSE_hierarchical-3d-gaussians.md (non-commercial research).
"""

import functools
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor"
GLM = VENDOR / "third_party"  # contains glm/
EIGEN = VENDOR / "third_party"  # contains Eigen/
GH = VENDOR / "gaussian_hierarchy"
HR = VENDOR / "hierarchy_rasterizer"


def _load(name, sources, includes, cuda_flags=()):
    from torch.utils.cpp_extension import load

    inc = [f"-I{p}" for p in includes]
    return load(name=name, sources=[str(s) for s in sources], extra_cflags=["-O3", *inc],
                extra_cuda_cflags=["-O3", *inc, *cuda_flags], verbose=False)


@functools.cache
def rasterizer():
    """diff_gaussian_rasterization._C of the hierarchy rasterizer."""
    src = [HR / "cuda_rasterizer" / f for f in ("rasterizer_impl.cu", "forward.cu", "backward.cu")]
    src += [HR / "rasterize_points.cu", HR / "ext.cpp"]
    return _load("ss_h3dgs_rasterizer", src, [GLM], ["-Xcompiler", "-fno-gnu-unique"])


@functools.cache
def hierarchy():
    """gaussian_hierarchy._C (same source list as the official setup.py)."""
    src = [GH / f for f in ("hierarchy_loader.cpp", "hierarchy_writer.cpp", "traversal.cpp",
                            "runtime_switching.cu", "torch/torch_interface.cpp", "ext.cpp")]
    return _load("ss_h3dgs_hierarchy", src, [EIGEN, GH])


@functools.cache
def creator():
    """GaussianHierarchyCreator sources (CMakeLists.txt) + csrc/creator_ext.cpp binding."""
    src = [GH / f for f in ("PointbasedKdTreeGenerator.cpp", "ClusterMerger.cpp",
                            "appearance_filter.cpp", "AvgMerger.cpp", "writer.cpp", "loader.cpp",
                            "hierarchy_writer.cpp", "hierarchy_loader.cpp", "traversal.cpp",
                            "rotation_aligner.cpp", "FlatGenerator.cpp", "runtime_switching.cu")]
    src += [HERE / "csrc" / "creator_ext.cpp"]
    return _load("ss_h3dgs_creator", src, [EIGEN, GH])
