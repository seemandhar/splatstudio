# Spec-Gaussian — implementation spec

Paper: Yang et al., *Spec-Gaussian: Anisotropic View-Dependent Appearance for 3D Gaussian
Splatting*, NeurIPS 2024, arXiv 2402.15870 (LaTeX v3 in `source/latex`).
Official code: github.com/ingra14m/Specular-Gaussians @0b40ce3 (MIT), `source/code`
(full history `source/code-full`); rasterizer submodule ingra14m/diff-gaussian-rasterization-extentions
@d595eac (`source/raster`).

## 1. Summary / contributions
- Replace pure SH colour by an **ASG appearance field**: c = c_d (SH, deg 3) + c_s, where c_s is
  decoded by a tiny MLP Ψ from a latent feature κ made of N anisotropic spherical Gaussians whose
  axes are fixed and whose λ, μ, ξ are predicted by a linear layer Θ from a per-Gaussian feature f.
- **C1 ASG appearance field** (§3.2, Eq. 5–8): per-Gaussian feature f, Θ(f) → {λ, μ, ξ}_N, κ = ⊕ ASG(ω | ...),
  Ψ(κ, γ(d), ⟨n,−d⟩) → c_s, c = c_d + c_s.
- **C2 normal / reflection** (§3.2): n = shortest axis, flipped towards camera; ω_r = 2(ω_o·n)n − ω_o.
- **C3 coarse-to-fine training** (§3.3): (a) L1 ("absolute") gradient accumulation for
  densification, τ_g = 0.0005 (performance) / 0.0006 (light); (b) resolution schedule
  r(i) = min(⌊r_s + (r_e − r_s)·i/τ⌉, r_e), r_s = 4× downsampled, τ = 5k (real scenes only).
- C4 anchor (Scaffold-GS) variant (§3.4) — "Ours-w/ anchor", not the headline row.

## 2. Equations (verbatim)
- Eq. ASG: $ASG(\nu | [x,y,z],[\lambda,\mu],\xi)=\xi\cdot S(\nu;z)\cdot e^{-\lambda(\nu\cdot x)^2-\mu(\nu\cdot y)^2}$, $S=\max(\nu\cdot z,0)$, ξ ∈ R².
- Eq. 6: $\Theta(\mathbf{f}) \rightarrow \{\lambda, \mu, \xi\}_N$
- Eq. 7: $c = c_d + c_s$
- Eq. 8: $\Psi(\kappa,\gamma(\mathbf d),\langle n,-\mathbf d\rangle)\to c_s,\ \kappa=\bigoplus_{i=1}^N ASG(\omega_r|[x,y,z],[\lambda_i,\mu_i],\xi_i)$
- Reflection: $\omega_r = 2(\omega_o\cdot n)n-\omega_o$, $\omega_o=-\mathbf d$.
- c2f gradient: $\frac{dL}{dx}=\sum\Vert\frac{dL}{dp_i}\frac{dp_i}{dx}\Vert_1$
- c2f resolution: $r(i)=\min(\lfloor r_s+(r_e-r_s)\cdot i/\tau\rceil, r_e)$
- Loss: $(1-\lambda)L_1+\lambda L_{D-SSIM}$, λ = 0.2.

## 3. What the official real-world code actually does (train.py, `--is_real`, `--asg_degree 12`)
The code is authoritative for the reported numbers (README: run_wo_anchor.sh "used to generate
the table in the paper"). It differs from the paper text:

| item | paper text | official code (Mip-NeRF 360) | source |
|---|---|---|---|
| per-Gaussian feature dim | 24 | **12** (`--asg_degree 12`) | run_wo_anchor.sh; gaussian_model.py:create_from_pcd |
| number of ASGs N | 32 | **8** (num_theta=2 × num_phi=4) | spec_utils.py:SpecularNetworkReal |
| ASG query direction | reflect ω_r | **view dir d = (x − cam)/‖·‖** (normal computed but unused) | ASGRenderReal.forward |
| Ψ input | κ, γ(d), ⟨n,−d⟩ | κ (16 = 8 ASG × ξ∈R²), d (3), PE(d) order 2 (12) → 31 | ASGRenderReal |
| Ψ layers | 3 × 64 | hidden 32; **outdoor: Linear-ReLU-Linear; indoor (`--is_indoor`): Linear-ReLU-Linear-ReLU-Linear**; last bias 0 | ASGRenderReal |
| Θ | MLP | single Linear(12 → 8·4) | SpecularNetworkReal.gaussian_feature |
| ASG params | λ, μ > 0 | λ = softplus(raw−1), μ = softplus(raw−1), ξ raw (2 ch) | RenderingEquationEncoding |
| resolution c2f | yes (real scenes) | **absent from train.py in all commits** (only train_anchor.py `--use_c2f`, 0.125→1 over 20k) | git log -S |
| abs-grad densify | yes | yes: CUDA `fabsf` per pixel and component (backward.cu:550) | raster |
| spec colour start | – | c_s = 0 for iteration ≤ 3000 | train.py |
| colour | c_d + c_s | `clamp_min(SH(d)+0.5, 0) + c_s` (no clamp after the sum) | gaussian_renderer/__init__.py:render |

Axes: `init_predefined_omega(2,4)`: θ ∈ linspace(0,π,2), φ ∈ linspace(0,2π,4); z = sph(θ,φ),
x = sph(θ+π/2, φ), y = rotate x about z by π/2 (quaternion). Fixed buffers.

## 4. Hyper-parameters (official train.py defaults; Mip-NeRF 360)
| name | value | source |
|---|---|---|
| iterations | 30000 | arguments |
| position lr | 1.6e-4·5 → 1.6e-6·5 (exp, delay_mult 0.01, no delay steps) | `spatial_lr_scale = 5` hard-coded (gaussian_model.py:training_setup) — not camera extent |
| f_dc / f_rest lr | 0.0025 / 0.0025/20 | |
| opacity / scaling / rotation lr | 0.05 / 0.005 / 0.001 | |
| f_asg lr | 0.0025 (constant), init 0, dim 12 | |
| Ψ/Θ (specular MLP) lr | linear 0.0025 → 0.000125 over 30000 (`get_linear_noise_func`), Adam eps 1e-15 | specular_model.py |
| densify | from 500 (`>`), every 100, until 15000, abs-grad threshold 0.0005, percent_dense 0.01, min opacity 0.005, max screen 20 after 3000, world 0.1·extent | train.py |
| opacity reset | every 3000 | |
| SH | degree 3, +1 every 1000 | |
| λ_D-SSIM | 0.2 | |
| background | black | |
| init opacity | 0.1 | |

## 5. Schedules
SH degree +1 / 1000 its; densification 500–15000; opacity reset /3000; specular branch on for
iteration > 3000; MLP lr linear decay; xyz lr exp decay; (paper only) resolution 1/4 → 1 over 5k.

## 6. Builds on
3DGS (vanilla train loop, densification, losses unchanged) + AbsGS-style abs gradient.

## 7. Protocol
Mip-NeRF 360, 9 scenes, outdoor `-r 4`, indoor `-r 2` (3DGS PIL resize of the full-res images →
our `mipnerf360-pilr` conversion), `--eval` (llffhold 8), 30k its, final iteration rendered
(render.py, iteration −1), metrics.py LPIPS-VGG. Per-scene PSNR/SSIM/LPIPS in supp tables.

## 8. Ambiguities
1. Paper text (N=32, f∈R²⁴, reflection query, 3×64 MLP) vs real-scene code (N=8, f∈R¹², view-dir
   query, 2/3-layer 32-wide). We follow the code (it produced the table).
2. Resolution coarse-to-fine: in the paper, absent in the code path that produced "Ours". Default
   follows code; `-c2f` variants implement the paper's schedule (r_s = 1/4, τ = 5000).
3. Indoor vs outdoor decoder depth is a per-scene flag → two method variants.
4. Paper's 3D-GS row (27.79) is stronger than the 3DGS paper (27.21); provenance unstated.
