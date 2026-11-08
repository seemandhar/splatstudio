/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "backward.h"
#include "auxiliary.h"
#include "stopthepop/resorted_render.cuh"
#include "stopthepop/hierarchical_render.cuh"
#include "stopthepop/consistent_common.cuh"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const bool proper_ewa_scaling,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* opacities,
	const float* dL_dconics,
	float* dL_dopacity,
	float3* dL_dmeans,
	float* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float c_xx = cov2D[0][0];
	float c_xy = cov2D[0][1];
	float c_yy = cov2D[1][1];
	const float det_cov_orig = c_xx * c_yy - c_xy * c_xy;

	constexpr float h_var = 0.3f;
	c_xx += h_var;
	c_yy += h_var;

	float dL_dc_xx = 0;
	float dL_dc_xy = 0;
	float dL_dc_yy = 0;
	if (proper_ewa_scaling)
	{
		// As employed by Yu et al. in "Mip-Splatting: Alias-free 3D Gaussian Splatting"
		// https://github.com/autonomousvision/mip-splatting
		const float det_cov_plus_h_cov = c_xx * c_yy - c_xy * c_xy;
		const float h_convolution_scaling = sqrt(max(0.000025f, det_cov_orig / det_cov_plus_h_cov)); // max for numerical stability
		const float dL_dopacity_v = dL_dopacity[idx];
		const float d_h_convolution_scaling = dL_dopacity_v * opacities[idx];
		dL_dopacity[idx] = dL_dopacity_v * h_convolution_scaling;
		const float d_inside_root = (det_cov_orig / det_cov_plus_h_cov) <= 0.000025f ? 0.f : d_h_convolution_scaling / (2 * h_convolution_scaling);

		// https://www.wolframalpha.com/input?i=d+%28%28x*y+-+z%5E2%29%2F%28%28x%2Bw%29*%28y%2Bw%29+-+z%5E2%29%29+%2Fdx
		// https://www.wolframalpha.com/input?i=d+%28%28x*y+-+z%5E2%29%2F%28%28x%2Bw%29*%28y%2Bw%29+-+z%5E2%29%29+%2Fdz
		const float x = c_xx;
		const float y = c_yy;
		const float z = c_xy;
		const float w = h_var;
		const float denom_f = d_inside_root / sq(w * w + w * (x + y) + x * y - z * z);
		const float dL_dx = w * (w * y + y * y + z * z) * denom_f;
		const float dL_dy = w * (w * x + x * x + z * z) * denom_f;
		const float dL_dz = -2.f * w * z * (w + x + y) * denom_f;
		dL_dc_xx = dL_dx;
		dL_dc_yy = dL_dy;
		dL_dc_xy = dL_dz;
	}

	float denom = c_xx * c_yy - c_xy * c_xy;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_dc_xx += denom2inv * (-c_yy * c_yy * dL_dconic.x + 2 * c_xy * c_yy * dL_dconic.y + (denom - c_xx * c_yy) * dL_dconic.z);
		dL_dc_yy += denom2inv * (-c_xx * c_xx * dL_dconic.z + 2 * c_xx * c_xy * dL_dconic.y + (denom - c_xx * c_yy) * dL_dconic.x);
		dL_dc_xy += denom2inv * 2 * (c_xy * c_yy * dL_dconic.x - (denom + 2 * c_xy * c_xy) * dL_dconic.y + c_xx * c_xy * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_dc_xx + T[0][0] * T[1][0] * dL_dc_xy + T[1][0] * T[1][0] * dL_dc_yy);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_dc_xx + T[0][1] * T[1][1] * dL_dc_xy + T[1][1] * T[1][1] * dL_dc_yy);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_dc_xx + T[0][2] * T[1][2] * dL_dc_xy + T[1][2] * T[1][2] * dL_dc_yy);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_dc_xx + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_dc_xy + 2 * T[1][0] * T[1][1] * dL_dc_yy;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_dc_xx + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_dc_xy + 2 * T[1][0] * T[1][2] * dL_dc_yy;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_dc_xx + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_dc_xy + 2 * T[1][1] * T[1][2] * dL_dc_yy;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_dc_xx +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc_xy;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_dc_xx +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc_xy;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_dc_xx +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc_xy;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc_yy +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_dc_xy;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc_yy +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_dc_xy;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc_yy +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_dc_xy;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = quat2mat(rot);
	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

__device__ void computeGauss2Screen_ProperEWA(int idx, int W, int H, float filter3D, float* dL_dopacity, const float* proj, const glm::vec3 scale, const glm::vec4 rot, const float* opacities, const glm::vec3& o, const float* dL_dgauss2screen, glm::vec3* dL_dmean, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	float variance_scale = filter3D * filter3D;
	//float variance_scale = 0.0f;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y + r * z), 2.f * (x * z - r * y),
		2.f * (x * y - r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + r * x),
		2.f * (x * z + r * y), 2.f * (y * z - r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::vec3 scale_squared = scale * scale;

	glm::vec3 scale_squared_dil = scale_squared;
	scale_squared_dil.x += variance_scale;
	scale_squared_dil.y += variance_scale;
	scale_squared_dil.z += variance_scale;

	glm::vec3 d = glm::normalize(o);
	glm::vec3 Rd = glm::transpose(R) * d;
	glm::vec3 Rd_squared = Rd * Rd;

	// A = variance along ray, B = variance along ray dilated
	float A = Rd_squared[0] * scale_squared[1] * scale_squared[2] + Rd_squared[1] * scale_squared[0] * scale_squared[2] + Rd_squared[2] * scale_squared[0] * scale_squared[1];
	float B = Rd_squared[0] * scale_squared_dil[1] * scale_squared_dil[2] + Rd_squared[1] * scale_squared_dil[0] * scale_squared_dil[2] + Rd_squared[2] * scale_squared_dil[0] * scale_squared_dil[1];

	float convolution_scale_factor = sqrtf(A / B);
	//float convolution_scale_factor = 1.0f;

	glm::vec3 s = glm::sqrt(scale_squared_dil);

	// Gradients w.r.t T
	const glm::mat4 viewport_mat = glm::transpose(glm::mat4(
		W / 2.0f, 0, 0, W / 2.0f - 0.5f,
		0, H / 2.0f, 0, H / 2.0f - 0.5f,
		0, 0, 1, 0,
		0, 0, 0, 1));
	glm::mat4 world2screen_mat = viewport_mat * loadMatrix4x4(proj);
	glm::mat4 dL_dT_prime_t = loadMatrix4x4(dL_dgauss2screen + idx * 16);
	glm::mat4 dL_dT = glm::transpose(world2screen_mat) * dL_dT_prime_t;

	glm::mat3 dL_dRS = glm::mat3(
		glm::vec3(dL_dT[0]),
		glm::vec3(dL_dT[1]),
		glm::vec3(dL_dT[2])
	);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;

	float dL_do = dL_dopacity[idx];
	// float dL_dconv = A < B ? dL_do * opacities[idx] * convolution_scale_factor / (A * B) : 0.0f;
	float dL_dconv = dL_do * opacities[idx] * convolution_scale_factor / (A * B);

	glm::vec3 dA_ds, dB_ds;

	dA_ds.x = scale.x * (Rd_squared[1] * scale_squared[2] + Rd_squared[2] * scale_squared[1]);
	dA_ds.y = scale.y * (Rd_squared[0] * scale_squared[2] + Rd_squared[2] * scale_squared[0]);
	dA_ds.z = scale.z * (Rd_squared[0] * scale_squared[1] + Rd_squared[1] * scale_squared[0]);

	dB_ds.x = scale.x * (Rd_squared[1] * scale_squared_dil[2] + Rd_squared[2] * scale_squared_dil[1]);
	dB_ds.y = scale.y * (Rd_squared[0] * scale_squared_dil[2] + Rd_squared[2] * scale_squared_dil[0]);
	dB_ds.z = scale.z * (Rd_squared[0] * scale_squared_dil[1] + Rd_squared[1] * scale_squared_dil[0]);

	dL_dscale->x = scale.x * glm::dot(R[0], dL_dRS[0]) / s.x + dL_dconv * (B * dA_ds.x - A * dB_ds.x);
	dL_dscale->y = scale.y * glm::dot(R[1], dL_dRS[1]) / s.y + dL_dconv * (B * dA_ds.y - A * dB_ds.y);
	dL_dscale->z = scale.z * glm::dot(R[2], dL_dRS[2]) / s.z + dL_dconv * (B * dA_ds.z - A * dB_ds.z);

	// Gradients of loss w.r.t opacity
	dL_dopacity[idx] = dL_do * convolution_scale_factor;
	
	// Gradients of loss w.r.t Rotation
	glm::vec3 dA_dRd = glm::vec3(Rd[0] * scale_squared[1] * scale_squared[2], Rd[1] * scale_squared[0] * scale_squared[2], Rd[2] * scale_squared[0] * scale_squared[1]);
	glm::vec3 dB_dRd = glm::vec3(Rd[0] * scale_squared_dil[1] * scale_squared_dil[2], Rd[1] * scale_squared_dil[0] * scale_squared_dil[2], Rd[2] * scale_squared_dil[0] * scale_squared_dil[1]);

	glm::vec3 dL_dRd = dL_dconv * (B * dA_dRd - A * dB_dRd);

	dL_dRS[0] = dL_dRS[0] * s.x + d * dL_dRd.x;
	dL_dRS[1] = dL_dRS[1] * s.y + d * dL_dRd.y;
	dL_dRS[2] = dL_dRS[2] * s.z + d * dL_dRd.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2.0f * z * (dL_dRS[0][1] - dL_dRS[1][0]) + 2.0f * y * (dL_dRS[2][0] - dL_dRS[0][2]) + 2.0f * x * (dL_dRS[1][2] - dL_dRS[2][1]);
	dL_dq.y = 2.0f * y * (dL_dRS[1][0] + dL_dRS[0][1]) + 2.0f * z * (dL_dRS[2][0] + dL_dRS[0][2]) + 2.0f * r * (dL_dRS[1][2] - dL_dRS[2][1]) - 4.0f * x * (dL_dRS[2][2] + dL_dRS[1][1]);
	dL_dq.z = 2.0f * x * (dL_dRS[1][0] + dL_dRS[0][1]) + 2.0f * r * (dL_dRS[2][0] - dL_dRS[0][2]) + 2.0f * z * (dL_dRS[1][2] + dL_dRS[2][1]) - 4.0f * y * (dL_dRS[2][2] + dL_dRS[0][0]);
	dL_dq.w = 2.0f * r * (dL_dRS[0][1] - dL_dRS[1][0]) + 2.0f * x * (dL_dRS[2][0] + dL_dRS[0][2]) + 2.0f * y * (dL_dRS[1][2] + dL_dRS[2][1]) - 4.0f * z * (dL_dRS[1][1] + dL_dRS[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });

	// Gradients of loss w.r.t 3d mean

	glm::mat3 dd_do = (glm::mat3(1.0f) - glm::outerProduct(d, d)) / glm::length(o);

	glm::vec3 dL_dd =  R * dL_dRd;

	dL_dmean[idx] = glm::vec3(dL_dT[3]) + dd_do * dL_dd;
}

__device__ void computeGauss2Screen(int idx, int W, int H, float filter3D, const float* proj, const glm::vec3 scale, const glm::vec4 rot, const float* dL_dgauss2screen, glm::vec3* dL_dmean, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	float variance_scale = filter3D * filter3D;
	//float variance_scale = 0.0f;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y + r * z), 2.f * (x * z - r * y),
		2.f * (x * y - r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + r * x),
		2.f * (x * z + r * y), 2.f * (y * z - r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::vec3 scale_squared = scale * scale;

	glm::vec3 scale_squared_dil = scale_squared;
	scale_squared_dil.x += variance_scale;
	scale_squared_dil.y += variance_scale;
	scale_squared_dil.z += variance_scale;

	glm::vec3 s = glm::sqrt(scale_squared_dil);

	// Gradients w.r.t T
	const glm::mat4 viewport_mat = glm::transpose(glm::mat4(
		W / 2.0f, 0, 0, W / 2.0f - 0.5f,
		0, H / 2.0f, 0, H / 2.0f - 0.5f,
		0, 0, 1, 0,
		0, 0, 0, 1));
	glm::mat4 world2screen_mat = viewport_mat * loadMatrix4x4(proj);
	glm::mat4 dL_dT_prime_t = loadMatrix4x4(dL_dgauss2screen + idx * 16);
	glm::mat4 dL_dT = glm::transpose(world2screen_mat) * dL_dT_prime_t;

	glm::mat3 dL_dRS = glm::mat3(
		glm::vec3(dL_dT[0]),
		glm::vec3(dL_dT[1]),
		glm::vec3(dL_dT[2])
	);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;

	dL_dscale->x = scale.x * glm::dot(R[0], dL_dRS[0]) / s.x;
	dL_dscale->y = scale.y * glm::dot(R[1], dL_dRS[1]) / s.y;
	dL_dscale->z = scale.z * glm::dot(R[2], dL_dRS[2]) / s.z;
	
	// Gradients of loss w.r.t Rotation
	dL_dRS[0] = dL_dRS[0] * s.x;
	dL_dRS[1] = dL_dRS[1] * s.y;
	dL_dRS[2] = dL_dRS[2] * s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2.0f * z * (dL_dRS[0][1] - dL_dRS[1][0]) + 2.0f * y * (dL_dRS[2][0] - dL_dRS[0][2]) + 2.0f * x * (dL_dRS[1][2] - dL_dRS[2][1]);
	dL_dq.y = 2.0f * y * (dL_dRS[1][0] + dL_dRS[0][1]) + 2.0f * z * (dL_dRS[2][0] + dL_dRS[0][2]) + 2.0f * r * (dL_dRS[1][2] - dL_dRS[2][1]) - 4.0f * x * (dL_dRS[2][2] + dL_dRS[1][1]);
	dL_dq.z = 2.0f * x * (dL_dRS[1][0] + dL_dRS[0][1]) + 2.0f * r * (dL_dRS[2][0] - dL_dRS[0][2]) + 2.0f * z * (dL_dRS[1][2] + dL_dRS[2][1]) - 4.0f * y * (dL_dRS[2][2] + dL_dRS[0][0]);
	dL_dq.w = 2.0f * r * (dL_dRS[0][1] - dL_dRS[1][0]) + 2.0f * x * (dL_dRS[2][0] + dL_dRS[0][2]) + 2.0f * y * (dL_dRS[1][2] + dL_dRS[2][1]) - 4.0f * z * (dL_dRS[1][1] + dL_dRS[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });

	// Gradients of loss w.r.t 3d mean
	dL_dmean[idx] = glm::vec3(dL_dT[3]);
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA_3D(
	int P, int D, int M, int W, int H, float focal_x,
	const bool proper_ewa,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float* opacities,
	const float scale_modifier,
	const float* viewmatrix,
	const float* proj,
	const glm::vec3* campos,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dL_dopacity,
	float* dL_dgauss2screen)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
	{
		glm::mat4x3 viewmat = loadMatrix4x3(viewmatrix);
		glm::vec3 mean = ((const glm::vec3*)means)[idx];
		glm::vec3 mean_view = viewmat * glm::vec4(mean, 1.0f);
		float filter_3D = fabsf(mean_view.z) / focal_x * sqrtf(0.3f);
		if (proper_ewa)
		{
			glm::vec3 ray = mean - *campos;
			computeGauss2Screen_ProperEWA(idx, W, H, filter_3D, dL_dopacity, proj, scales[idx], rotations[idx], opacities, ray, dL_dgauss2screen, dL_dmeans, dL_dscale, dL_drot);
		}
		else
			computeGauss2Screen(idx, W, H, filter_3D, proj, scales[idx], rotations[idx], dL_dgauss2screen, dL_dmeans, dL_dscale, dL_drot);
	}
}

// Backward version of the rendering procedure.
template <uint32_t C, bool EVAL_3D>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ gauss2screen,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float4* __restrict__ dL_dgauss2screen)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_opacity[BLOCK_SIZE];

	__shared__ float collected_colors[C * BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			if constexpr (EVAL_3D)
			{
				collected_opacity[block.thread_rank()] = ((float*) conic_opacity)[coll_id];
			}
			else
			{
				collected_xy[block.thread_rank()] = points_xy_image[coll_id];
				collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			}		
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			float2 d;
			float4 con_o;
			glm::mat4 gauss2screen_mat;
			float alpha, G;

			if constexpr (EVAL_3D)
			{
				con_o.w = collected_opacity[j];
				gauss2screen_mat = loadMatrix4x4(gauss2screen + collected_id[j] * 16);

				const glm::vec4 plane_x = gauss2screen_mat[0] - gauss2screen_mat[3] * pixf.x;
				const glm::vec4 plane_y = gauss2screen_mat[1] - gauss2screen_mat[3] * pixf.y;
				
				glm::vec4 max_pos;
				float power = -0.5f * max_contrib_ray(plane_x, plane_y, max_pos);

				G = exp(power);
				alpha = min(0.99f, con_o.w * G);
				if (alpha < ALPHA_THRESHOLD)
					continue;
				
				float depth = glm::dot(gauss2screen_mat[2], max_pos) / glm::dot(gauss2screen_mat[3], max_pos);
				if (depth < -1.0f || depth > 1.0f)
					continue;
			}
			else
			{
				con_o = collected_conic_opacity[j];
				const float2 xy = collected_xy[j];

				d = { xy.x - pixf.x, xy.y - pixf.y };
				float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;

				if (power > 0.0f)
					continue;

				G = exp(power);
				alpha = min(0.99f, con_o.w * G);
				if (alpha < ALPHA_THRESHOLD)
					continue;
			}

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;

			if constexpr (EVAL_3D)
			{
				const glm::vec4 p_x = { 1.0f, 0.0f, 0.0f, -pixf.x };
				const glm::vec4 p_y = { 0.0f, 1.0f, 0.0f, -pixf.y };

				const glm::vec4 plane_x = gauss2screen_mat[0] - gauss2screen_mat[3] * pixf.x;
				const glm::vec4 plane_y = gauss2screen_mat[1] - gauss2screen_mat[3] * pixf.y;

				const glm::vec3 n_x = plane_x; // xyz
				const glm::vec3 n_y = plane_y; // xyz
				const float d_x = plane_x.w;
				const float d_y = plane_y.w;

				const glm::vec3 m = d_x * n_y - d_y * n_x;
				const glm::vec3 d = glm::cross(n_x, n_y);
				const float d_2 = glm::dot(d, d);
				const float m_2 = glm::dot(m, m);
				const float n_x_2 = glm::dot(n_x, n_x);
				const float n_y_2 = glm::dot(n_y, n_y);
				const float n_xy = glm::dot(n_x, n_y);
				const float n_y_m = glm::dot(n_y, m);
				const float n_x_m = glm::dot(n_x, m);

				glm::vec3 dd_dT0 = n_y_2 * n_x - n_xy * n_y;
				glm::vec3 dd_dT1 = n_x_2 * n_y - n_xy * n_x;
				glm::vec3 dd_dT3 = n_xy * (pixf.y * n_x + pixf.x * n_y) - pixf.x * n_y_2 * n_x - pixf.y * n_x_2 * n_y;
				glm::vec4 dd_dT_u = { dd_dT0[0], dd_dT1[0], 0.0f, dd_dT3[0] };
				glm::vec4 dd_dT_v = { dd_dT0[1], dd_dT1[1], 0.0f, dd_dT3[1] };
				glm::vec4 dd_dT_w = { dd_dT0[2], dd_dT1[2], 0.0f, dd_dT3[2] };

				glm::vec3 dm_dT0 = -d_y * m;
				glm::vec3 dm_dT1 = d_x * m;
				glm::vec3 dm_dT3 = (d_y * pixf.x - d_x * pixf.y) * m;
				glm::vec4 dm_dT_u = { dm_dT0[0], dm_dT1[0], 0.0f, dm_dT3[0] };
				glm::vec4 dm_dT_v = { dm_dT0[1], dm_dT1[1], 0.0f, dm_dT3[1] };
				glm::vec4 dm_dT_w = { dm_dT0[2], dm_dT1[2], 0.0f, dm_dT3[2] };

				glm::vec4 dm_dT_mu = { n_y_m, -n_x_m, 0.0f,  pixf.y * n_x_m - pixf.x * n_y_m };

				glm::vec4 dG_dT_u = dL_dG * G * (m_2 / d_2 * __frcp_rn(d_2) * dd_dT_u - dm_dT_u / d_2);
				glm::vec4 dG_dT_v = dL_dG * G * (m_2 / d_2 * __frcp_rn(d_2) * dd_dT_v - dm_dT_v / d_2);
				glm::vec4 dG_dT_w = dL_dG * G * (m_2 / d_2 * __frcp_rn(d_2) * dd_dT_w - dm_dT_w / d_2);

				glm::vec4 dG_dT_mu = dL_dG * G * -dm_dT_mu / d_2;

				int global_id4 = global_id * 4;
				// gradient w.r.t z is always 0!
				atomicAdd(&dL_dgauss2screen[global_id4].x, dG_dT_u.x);
				atomicAdd(&dL_dgauss2screen[global_id4].y, dG_dT_u.y);
				atomicAdd(&dL_dgauss2screen[global_id4].w, dG_dT_u.w);

				atomicAdd(&dL_dgauss2screen[global_id4 + 1].x, dG_dT_v.x);
				atomicAdd(&dL_dgauss2screen[global_id4 + 1].y, dG_dT_v.y);
				atomicAdd(&dL_dgauss2screen[global_id4 + 1].w, dG_dT_v.w);

				atomicAdd(&dL_dgauss2screen[global_id4 + 2].x, dG_dT_w.x);
				atomicAdd(&dL_dgauss2screen[global_id4 + 2].y, dG_dT_w.y);
				atomicAdd(&dL_dgauss2screen[global_id4 + 2].w, dG_dT_w.w);

				atomicAdd(&dL_dgauss2screen[global_id4 + 3].x, dG_dT_mu.x);
				atomicAdd(&dL_dgauss2screen[global_id4 + 3].y, dG_dT_mu.y);
				atomicAdd(&dL_dgauss2screen[global_id4 + 3].w, dG_dT_mu.w);
			}
			else
			{
				const float gdx = G * d.x;
				const float gdy = G * d.y;
				const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
				const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;
	
				// Update gradients w.r.t. 2D mean position of the Gaussian
				atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
				atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);
	
				// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
				atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
				atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
				atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);
			}

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M, int W, int H,
	const bool eval_3D,
	const bool proper_ewa_scaling,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const float* opacities,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	float* dL_dopacity,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dL_dgauss2screen)
{
	if (!eval_3D)
	{
		// Propagate gradients for the path of 2D conic matrix computation. 
		// Somewhat long, thus it is its own kernel rather than being part of 
		// "preprocess". When done, loss gradient w.r.t. 3D means has been
		// modified and gradient w.r.t. 3D covariance matrix has been computed.	
		computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
			P,
			proper_ewa_scaling,
			means3D,
			radii,
			cov3Ds,
			focal_x,
			focal_y,
			tan_fovx,
			tan_fovy,
			viewmatrix,
			opacities,
			dL_dconic,
			dL_dopacity,
			(float3*)dL_dmean3D,
			dL_dcov3D);
	}

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	if (eval_3D)
	{
		preprocessCUDA_3D<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
			P, D, M, W, H, focal_x, proper_ewa_scaling,
			(float3*)means3D,
			radii,
			shs,
			clamped,
			(glm::vec3*)scales,
			(glm::vec4*)rotations,
			opacities,
			scale_modifier,
			viewmatrix,
			projmatrix,
			campos,
			(glm::vec3*)dL_dmean3D,
			dL_dcolor,
			dL_dsh,
			dL_dscale,
			dL_drot,
			dL_dopacity,
			dL_dgauss2screen);
	}
	else
	{
		preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
			P, D, M,
			(float3*)means3D,
			radii,
			shs,
			clamped,
			(glm::vec3*)scales,
			(glm::vec4*)rotations,
			scale_modifier,
			projmatrix,
			campos,
			(float3*)dL_dmean2D,
			(glm::vec3*)dL_dmean3D,
			dL_dcolor,
			dL_dcov3D,
			dL_dsh,
			dL_dscale,
			dL_drot);
	}

	#undef PREPROCESS
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	SortSettings sort_settings,
	CullingSettings culling_settings,
	const uint32_t* point_list,
	int W, int H,
	const bool eval_3D,
	const float* gauss2screen,
	const float* bg_color,
	const float2* means2D,
	const float4* cov3D_inv,
	const float* projmatrix_inv,
	const glm::vec3* cam_pos,
	const float4* conic_opacity,
	const float* colors,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* pixel_colors,
	const float* dL_dpixels,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float4* dL_dgauss2screen)
{
	#define CALL_KBUFFER(WINDOW) renderkBufferBackwardCUDA<NUM_CHANNELS, WINDOW> << <grid, block >> > (ranges, point_list, W, H, means2D, cov3D_inv, projmatrix_inv, (float3*)cam_pos, colors, conic_opacity, final_Ts, n_contrib, bg_color, pixel_colors, dL_dpixels, dL_dmean2D, dL_dconic2D, dL_dopacity, dL_dcolors)

	if (sort_settings.sort_mode == SortMode::GLOBAL)
	{
		#define CALL_VANILLA(EVAL_3D) \
		renderCUDA<NUM_CHANNELS, EVAL_3D> << <grid, block >> >( \
			ranges, \
			point_list, \
			W, H, \
			gauss2screen, \
			bg_color, \
			means2D, \
			conic_opacity, \
			colors, \
			final_Ts, \
			n_contrib, \
			dL_dpixels, \
			dL_dmean2D, \
			dL_dconic2D, \
			dL_dopacity, \
			dL_dcolors, \
			dL_dgauss2screen \
			);

		if (eval_3D)
			CALL_VANILLA(true)
		else
			CALL_VANILLA(false)

		#undef CALL_VANILLA
	}
	else if (sort_settings.sort_mode == SortMode::PER_PIXEL_KBUFFER)
	{
#ifdef STOPTHEPOP_FASTBUILD
		CALL_KBUFFER(16);
#else // STOPTHEPOP_FASTBUILD
		int window_size = sort_settings.queue_sizes.per_pixel;
		if (window_size <= 1) 
			CALL_KBUFFER(1); 
		else if (window_size <= 2) 
			CALL_KBUFFER(2); 
		else if (window_size <= 4) 
			CALL_KBUFFER(4); 
		else if (window_size <= 8) 
			CALL_KBUFFER(8); 
		else if (window_size <= 12) 
			CALL_KBUFFER(12); 
		else if (window_size <= 16) 
			CALL_KBUFFER(16); 
		else if (window_size <= 20) 
			CALL_KBUFFER(20); 
		else 
			CALL_KBUFFER(24);
#endif // STOPTHEPOP_FASTBUILD
		
		#undef CALL_KBUFFER
		#undef CALL_KBUFFER_DEBUG
		return;
	}
	else if (sort_settings.sort_mode == SortMode::PER_PIXEL_FULL)
	{
		throw std::runtime_error("Backward not supported for full per-pixel sort");
	}
	else if (sort_settings.sort_mode == SortMode::HIERARCHICAL)
	{
#define CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE, EVAL_3D) \
	sortGaussiansRayHierarchicalCUDA_backward<NUM_CHANNELS, HEAD_QUEUE_SIZE, MID_QUEUE_SIZE, HIER_CULLING, EVAL_3D><<<grid, {16, 4, 4}>>>( \
		ranges, point_list, W, H, gauss2screen, bg_color, means2D, cov3D_inv, projmatrix_inv, (float3*) cam_pos, conic_opacity, \
		colors, final_Ts, n_contrib, pixel_colors, dL_dpixels, dL_dmean2D, dL_dconic2D, dL_dopacity, dL_dcolors, dL_dgauss2screen)

#define CALL_HIER_EVAL3D(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE) if (eval_3D) { CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE, true); } else { CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE, false); }

#ifdef STOPTHEPOP_FASTBUILD
#define CALL_HIER_HEAD(HIER_CULLING, MID_QUEUE_SIZE) \
	switch (sort_settings.queue_sizes.per_pixel) \
	{ \
		case 4: { CALL_HIER_EVAL3D(HIER_CULLING, MID_QUEUE_SIZE, 4); break; } \
		default: { throw std::runtime_error("Not supported head queue size"); } \
	}

#define CALL_HIER_MID(HIER_CULLING) \
	switch (sort_settings.queue_sizes.tile_2x2) \
	{ \
		case 8: { CALL_HIER_HEAD(HIER_CULLING, 8); break; } \
		default: { throw std::runtime_error("Not supported mid queue size"); } \
	}
#else // STOPTHEPOP_FASTBUILD
#define CALL_HIER_HEAD(HIER_CULLING, MID_QUEUE_SIZE) \
	switch (sort_settings.queue_sizes.per_pixel) \
	{ \
		case 4: { CALL_HIER_EVAL3D(HIER_CULLING, MID_QUEUE_SIZE, 4); break; } \
		case 8: { CALL_HIER_EVAL3D(HIER_CULLING, MID_QUEUE_SIZE, 8); break; } \
		case 12: { CALL_HIER_EVAL3D(HIER_CULLING, MID_QUEUE_SIZE, 12); break; } \
		case 16: { CALL_HIER_EVAL3D(HIER_CULLING, MID_QUEUE_SIZE, 16); break; } \
		default: { throw std::runtime_error("Not supported head queue size " + std::to_string(sort_settings.queue_sizes.per_pixel)); } \
	}

#define CALL_HIER_MID(HIER_CULLING) \
	switch (sort_settings.queue_sizes.tile_2x2) \
	{ \
		case 8: { CALL_HIER_HEAD(HIER_CULLING, 8); break; } \
		case 12: { CALL_HIER_HEAD(HIER_CULLING, 12); break; } \
		case 20: { CALL_HIER_HEAD(HIER_CULLING, 20); break; } \
		default: { throw std::runtime_error("Not supported mid queue size " + std::to_string(sort_settings.queue_sizes.tile_2x2)); } \
	}
#endif // STOPTHEPOP_FASTBUILD

	if (culling_settings.hierarchical_4x4_culling) {
		CALL_HIER_MID(true);
	} else {
		CALL_HIER_MID(false);
	}

#undef CALL_HIER_MID
#undef CALL_HIER_HEAD
#undef CALL_HIER_EVAL3D
#undef CALL_HIER
	}
}