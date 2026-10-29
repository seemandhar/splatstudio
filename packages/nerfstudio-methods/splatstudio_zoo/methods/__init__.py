"""
splatstudio_zoo.methods — Fully integrated example methods
=======================================================

Each method is a self-contained model + config demonstrating
the template at different complexity levels:

    absgs            ★☆☆☆☆  Config-only (absgrad=True)
    depth_reg        ★★☆☆☆  Custom loss function
    feature_gs       ★★★★☆  New params, rendering, loss, densification
    micro_splatting  ★★★☆☆  Custom losses + Stage II pruning/merging
"""

from splatstudio_zoo.methods.absgs import AbsGSModel, AbsGSModelConfig
from splatstudio_zoo.methods.depth_reg import DepthRegGSModel, DepthRegGSModelConfig
from splatstudio_zoo.methods.feature_gs import FeatureGSModel, FeatureGSModelConfig
from splatstudio_zoo.methods.micro_splatting import MicroSplattingModel, MicroSplattingModelConfig
from splatstudio_zoo.methods.compact_gs import CompactGSModel, CompactGSModelConfig
from splatstudio_zoo.methods.msaa_gs import MSAAGSModel, MSAAGSModelConfig
from splatstudio_zoo.methods.opti3dgs import Opti3DGSModel, Opti3DGSModelConfig
from splatstudio_zoo.methods.decomp_gs import DecompGSModel, DecompGSModelConfig
from splatstudio_zoo.methods.bogauss import BOGausSModel, BOGausSModelConfig
from splatstudio_zoo.methods.light_gs import LightGSModel, LightGSModelConfig
from splatstudio_zoo.methods.vod3dgs import VoD3DGSModel, VoD3DGSModelConfig
from splatstudio_zoo.methods.rain_gs import RAINGSModel, RAINGSModelConfig
from splatstudio_zoo.methods.trick_gs import TrickGSModel, TrickGSModelConfig
from splatstudio_zoo.methods.clod_gs import CLoDGSModel, CLoDGSModelConfig
from splatstudio_zoo.methods.twodgs import TwoDGSModel, TwoDGSModelConfig
from splatstudio_zoo.methods.pixel_gs import PixelGSModel, PixelGSModelConfig
from splatstudio_zoo.methods.gaussian_shader import GaussianShaderModel, GaussianShaderModelConfig
from splatstudio_zoo.methods.scaffold_gs import ScaffoldGSModel, ScaffoldGSModelConfig
from splatstudio_zoo.methods.dwtgs import DWTGSModel, DWTGSModelConfig
from splatstudio_zoo.methods.lp3dgs import LP3DGSModel, LP3DGSModelConfig
from splatstudio_zoo.methods.compgs import CompGSModel, CompGSModelConfig
from splatstudio_zoo.methods.compgs_enhanced import CompGSEnhancedModel, CompGSEnhancedModelConfig
from splatstudio_zoo.methods.cor_gs import CoRGSModel, CoRGSModelConfig
from splatstudio_zoo.methods.fsgs import FSGSModel, FSGSModelConfig
from splatstudio_zoo.methods.diamond_gs_3d import DiamondGS3DModel, DiamondGS3DModelConfig

__all__ = [
    "AbsGSModel",
    "AbsGSModelConfig",
    "DepthRegGSModel",
    "DepthRegGSModelConfig",
    "FeatureGSModel",
    "FeatureGSModelConfig",
    "MicroSplattingModel",
    "MicroSplattingModelConfig",
    "CompactGSModel",
    "CompactGSModelConfig",
    "MSAAGSModel",
    "MSAAGSModelConfig",
    "Opti3DGSModel",
    "Opti3DGSModelConfig",
    "DecompGSModel",
    "DecompGSModelConfig",
    "BOGausSModel",
    "BOGausSModelConfig",
    "LightGSModel",
    "LightGSModelConfig",
    "VoD3DGSModel",
    "VoD3DGSModelConfig",
    "RAINGSModel",
    "RAINGSModelConfig",
    "TrickGSModel",
    "TrickGSModelConfig",
    "CLoDGSModel",
    "CLoDGSModelConfig",
    "TwoDGSModel",
    "TwoDGSModelConfig",
    "PixelGSModel",
    "PixelGSModelConfig",
    "GaussianShaderModel",
    "GaussianShaderModelConfig",
    "ScaffoldGSModel",
    "ScaffoldGSModelConfig",
    "DWTGSModel",
    "DWTGSModelConfig",
    "LP3DGSModel",
    "LP3DGSModelConfig",
    "CompGSModel",
    "CompGSModelConfig",
    "CompGSEnhancedModel",
    "CompGSEnhancedModelConfig",
    "CoRGSModel",
    "CoRGSModelConfig",
    "FSGSModel",
    "FSGSModelConfig",
    "DiamondGS3DModel",
    "DiamondGS3DModelConfig",
]

