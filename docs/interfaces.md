# Runtime interfaces, version 1

`MethodSpec` contains named `ComponentSpec`, `FieldSpec`, `ProbeSpec`, and
`ContributionSpec` values. The public registry resolves Python entry points;
duplicate names fail instead of silently shadowing an implementation.

## Representation and units

The native v0.1 state is a `torch.nn.ParameterDict`. Each field has one Adam
optimizer. Base fields are `means [N,3]` in world units, `scales [N,3]` in log
world units, `quats [N,4]` in wxyz order, `opacities [N]` in logit space,
`sh0 [N,1,3]`, and `shN [N,15,3]`. The backend activates scale and opacity and
uses gsplat's quaternion handling. Extra `FieldSpec` values are Gaussian-owned;
they inherit the population indexing of these fields. Domain-specific transforms
that require something other than copying an attribute need a custom density
component. Anchor-owned and global learned fields are not supported yet.

Camera dictionaries contain an OpenCV world-to-camera `viewmat [4,4]`, pixel
intrinsics `K [3,3]`, and integer `width`/`height`. Renderer outputs use HWC RGB
over a white background and expected camera-z depth `[H,W,1]`. Camera depth is
not ray distance. The evaluator clamps RGB to [0,1]; training does not clamp it.
The native 2D renderer returns world-space normals, normals from depth, gsplat's
L1 distortion, and `gradient_2dgs` statistics. Its state retains three scale
components for backend compatibility. The runtime adds the current one-based
`step` to training outputs, allowing explicit loss schedules. The 2D recipe
uses distortion weight 0.01 after step 3,000 and normal weight 0.05 after step
7,000, following the [gsplat 1.5.3 example's scheduling conventions](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/examples/simple_trainer_2dgs.py).

## Execution

Factories receive their component's full specification. Native runtime order:

1. Initialize state, optimizers, and density controller.
2. Render one training camera.
3. Call `density.before_backward` to retain required intermediate gradients.
4. Evaluate additive losses, check scalar/finiteness/graph attachment, and backpropagate.
5. Apply optimizer steps and clear parameter gradients.
6. Call `density.after_backward`, then check population and optimizer alignment
   after population changes and at logging intervals.

The renderer must expose its actual `capabilities`; the density controller must
expose its actual `requires`. Their declarations and runtime values must match.
Static dependency resolution detects missing providers and circular dependencies.
These are interface checks. A dishonest or incorrect implementation can still
advertise capabilities it does not implement; executable probes supply additional
evidence and independent benchmark tests remain necessary.

## Evidence

A probe function `(MethodSpec, device) -> dict` executes checks and returns
measurements. Raise on failure, or raise `ProbeUnavailable` when a necessary
resource is absent. A contribution is checked only when every referenced probe
passes; an empty probe list is unverified. Reports never promote a passed probe
suite into a claim of paper reproduction.

The provided state probe uses gsplat's actual duplicate/split/prune operations,
with allocated Adam moments and custom fields. The renderer probe executes the
configured rasterizer and checks signed/absolute screen gradients. The scale-prior
example compares its implemented derivative to a separate analytic formula.
These probes are intentionally small. They do not establish convergence,
coverage of a complete method, or correctness on all inputs.
