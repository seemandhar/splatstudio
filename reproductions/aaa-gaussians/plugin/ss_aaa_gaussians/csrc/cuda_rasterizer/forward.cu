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

#include "forward.h"
#include "forward_common.h"
#include "auxiliary.h"
#include "stopthepop/consistent_common.cuh"
#include "stopthepop/stopthepop_common.cuh"
#include "stopthepop/resorted_render.cuh"
#include "stopthepop/hierarchical_render.cuh"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;


// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeysCUDA(
	int P,
	const float2* rects,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	const int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], rects[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint32_t tile_id = y * grid.x + x;
				gaussian_keys_unsorted[off] = constructSortKey(tile_id, depths[idx]);
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C, bool TILE_BASED_CULLING, bool LOAD_BALANCING, bool EVAL_3D>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* rects,
	const GlobalSortOrder sort_order,
	const bool rect_bounding,
	const bool tight_opacity_bounding,
	const bool proper_ewa_scaling,
	const bool new_aabb,
	const bool near_clipping,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float4* cov3D_invs,
	float* gauss2screen,
	float* filter_3D,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
#define RETURN_OR_INACTIVE() if constexpr(TILE_BASED_CULLING && LOAD_BALANCING) { active = false; } else { return; }

	auto idx = cg::this_grid().thread_rank();
	bool active = true;
	if (idx >= P) {
		RETURN_OR_INACTIVE();
		idx = P - 1;
	}

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	const glm::vec3 mean3D(orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]);
	float opacity = opacities[idx];

	const glm::mat4x3 viewmatrix_mat = loadMatrix4x3(viewmatrix);
	glm::vec3 mean3D_view = viewmatrix_mat * glm::vec4(mean3D, 1.0f);

 	float radius;
	float2 rect_dims;
	float2 mean2D;
	float opacity_power_threshold;

	bool valid = false;
	float4 co;
	glm::mat4 gauss2screen_mat;
	if constexpr (EVAL_3D)
	{
		if (near_clipping && mean3D_view.z < 0.2f)
			RETURN_OR_INACTIVE();

		const glm::mat4 viewproj_mat = loadMatrix4x4(projmatrix);
		const glm::mat4 viewport_mat = glm::transpose(glm::mat4(
			W / 2.0f, 0, 0, W / 2.0f - 0.5f,
			0, H / 2.0f, 0, H / 2.0f - 0.5f,
			0, 0, 1, 0,
			0, 0, 0, 1));

		const glm::mat4 world2screen_mat = viewport_mat * viewproj_mat;

		float dilation_factor = 1.0f;
		float kernel_size = 0.3f;

		glm::vec3 campos_gauss;
		glm::mat4 gauss2view_mat;
		gauss2screen_mat = compute_gauss2screen(mean3D, scales[idx], rotations[idx], scale_modifier,
												*cam_pos, world2screen_mat, viewmatrix_mat, max(focal_x, focal_y), kernel_size,
												gauss2view_mat, campos_gauss, dilation_factor, filter_3D ? filter_3D[idx] : 0.0f);

		opacity *= proper_ewa_scaling ? dilation_factor : 1.0f;
		if (opacity < ALPHA_THRESHOLD)
			RETURN_OR_INACTIVE();
		
		co = make_float4(0, 0, 0, opacity);
		opacity_power_threshold = log(opacity / ALPHA_THRESHOLD);
		const float cutoff = min(11.11f, 2.0f * opacity_power_threshold);

		float distance_campos = glm::dot(campos_gauss, campos_gauss);
		if (glm::dot(campos_gauss, campos_gauss) < cutoff) // Camera is inside the ellipsoid
			RETURN_OR_INACTIVE();

		float max_pos_depth;
		const glm::vec2 screen_min = glm::vec2(0.0, 0.0f);
		const glm::vec2 screen_max = glm::vec2(W-1.f, H-1.f);

		glm::vec2 mean2D_new, extent_new;
		if (false && glm::dot(campos_gauss, campos_gauss) < cutoff) // Camera is inside the ellipsoid
		{
			mean2D_new = (screen_max + screen_min) / 2.0f;
			extent_new = (screen_max - screen_min) / 2.0f;
		}
		else
		{
			float max_contrib = max_contrib_gaussian_frustum_3D(screen_min, screen_max, gauss2screen_mat, max_pos_depth);
			valid = max_contrib <= opacity_power_threshold;
			if (max_contrib > opacity_power_threshold)
				RETURN_OR_INACTIVE();

			if ((!new_aabb && active && !compute_aabb_screen(gauss2screen_mat, mean3D_view, glm::vec4(0, 1, 0, 0), focal_x, focal_y, W, H, cutoff, mean2D_new, extent_new)) || 
				(new_aabb && active && !compute_aabb_view(gauss2view_mat, mean3D_view, glm::vec4(0, 1, 0, 0), focal_x, focal_y, W, H, cutoff, mean2D_new, extent_new)))
			{
				RETURN_OR_INACTIVE();
			}
		}		
			
		mean2D = {mean2D_new.x, mean2D_new.y};
		rect_dims = {extent_new.x, extent_new.y};
		radius = max(extent_new.x, extent_new.y);

		if (active)
			storeMatrix4x4(gauss2screen_mat, gauss2screen + 16 * idx);
		((float*)conic_opacity)[idx] = opacity;
	}
	else
	{
		// Perform near culling, quit if outside.
		if (mean3D_view.z < 0.2f)
			RETURN_OR_INACTIVE();

		// If 3D covariance matrix is precomputed, use it, otherwise compute
		// from scaling and rotation parameters. 
		const float* cov3D;
		if (cov3D_precomp != nullptr)
		{
			cov3D = cov3D_precomp + idx * 6;
		}
		else
		{
			computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
			cov3D = cov3Ds + idx * 6;
		}

		// Compute 2D screen-space covariance matrix
		glm::mat3 cov = computeCov2D(mean3D_view, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix_mat);

		float det, convolution_scaling_factor;
		glm::vec3 cov2D = dilateCov2D(cov, proper_ewa_scaling, det, convolution_scaling_factor);
		if (det == 0.0f)
			RETURN_OR_INACTIVE();

		// Invert covariance (EWA algorithm)
		co = active ? computeConicOpacity(cov2D, opacity, det, convolution_scaling_factor) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

		if (co.w < ALPHA_THRESHOLD)
			RETURN_OR_INACTIVE();

		// Slightly higher threshold for tile-based culling; Otherwise, imprecisions could lead to more tiles in preprocess than in duplicate
		constexpr float alpha_threshold = TILE_BASED_CULLING ? ALPHA_THRESHOLD_PADDED : ALPHA_THRESHOLD;
		opacity_power_threshold = log(co.w / alpha_threshold);

		// Compute extent in screen space (by finding eigenvalues of 2D covariance matrix).
		const float extent = tight_opacity_bounding ? min(3.33, sqrt(2.0f * opacity_power_threshold)) : 3.33f;

		const float min_lambda = 0.01f;
		const float mid = 0.5f * (cov2D.x + cov2D.z);
		const float lambda = mid + sqrt(max(min_lambda, mid * mid - det));
		radius = extent * sqrt(lambda);

		if (radius <= 0.0f)
			RETURN_OR_INACTIVE();

		// Transform point by projecting
		const glm::mat4 viewproj_mat = loadMatrix4x4(projmatrix);
		const glm::vec3 p_proj = world2ndc(mean3D, viewproj_mat);
		mean2D = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };

		// Use extent to compute a bounding rectangle of screen-space tiles that this Gaussian overlaps with.
		// Quit if rectangle covers 0 tiles
		const float extent_x = min(rect_bounding ? (extent * sqrt(cov2D.x)) : radius, radius);
		const float extent_y = min(rect_bounding ? (extent * sqrt(cov2D.z)) : radius, radius);
		rect_dims = make_float2(extent_x, extent_y);
		conic_opacity[idx] = co; // Inverse 2D covariance and opacity neatly pack into one float4
	}

	uint2 rect_min, rect_max;
	getRect(mean2D, rect_dims, rect_min, rect_max, grid);

	const int tile_count_rect = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
	if (tile_count_rect == 0)
	{
		RETURN_OR_INACTIVE();
	}

	const uint32_t WARP_MASK = 0xFFFFFFFFU;
	if constexpr(TILE_BASED_CULLING && LOAD_BALANCING)
		if (__ballot_sync(WARP_MASK, active) == 0) // early stop if whole warp culled
			return;

	int tile_count;
	if constexpr (TILE_BASED_CULLING)
		tile_count = computeTilebasedCullingTileCount<LOAD_BALANCING, EVAL_3D>(active, idx, co, gauss2screen_mat,  mean2D, opacity_power_threshold, rect_min, rect_max, gauss2screen);
	else
		tile_count = tile_count_rect;

	if (tile_count == 0 || !active) // Cooperative threads no longer needed (after load balancing)
	{
		return;
	}

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, mean3D, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	if (!EVAL_3D && cov3D_invs != nullptr)
	{
		const glm::vec3 mean3D(orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]);
		glm::mat3 inv = computeInvCov3D(scales[idx], rotations[idx], scale_modifier);

		// symmetric matrix, store six elements 
		// pack with Cov3dinv*(campos - mean) into 3 float4 for efficiency
		// we do have 3 elements leftover
		glm::vec3 upper = -inv * (*cam_pos - mean3D);
		cov3D_invs[3 * idx] = { inv[0][0], inv[0][1], inv[0][2], 0 };
		cov3D_invs[3 * idx + 1] = { inv[1][1], inv[1][2], inv[2][2], 0 };
		cov3D_invs[3 * idx + 2] = { upper.x, upper.y, upper.z, 0 };
	}

	// Store some useful helper data for the next steps.
	depths[idx] = sort_order == GlobalSortOrder::VIEWSPACE_Z ? mean3D_view.z : glm::length(*cam_pos - mean3D);
	radii[idx] = (int) ceil(radius);
	rects[idx] = rect_dims;
	points_xy_image[idx] = mean2D;
	tiles_touched[idx] = tile_count;
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS, bool ENABLE_DEBUG_VIZ, bool EVAL_3D>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	DebugVisualization debugVisualizationType,
	const glm::vec3* cam_pos,
	const glm::vec3* means3D,
	const float* gauss2screen,
	float* __restrict__ out_color)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x , (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	const int THREADS_PER_GAUSSIAN = EVAL_3D ? 4 : 1;
	const int N_ENTRIES_PER_ROUND = BLOCK_SIZE / THREADS_PER_GAUSSIAN;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + N_ENTRIES_PER_ROUND - 1) / N_ENTRIES_PER_ROUND);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[N_ENTRIES_PER_ROUND];
	__shared__ float2 collected_xy[N_ENTRIES_PER_ROUND];
	__shared__ float4 collected_conic_opacity[N_ENTRIES_PER_ROUND];
	__shared__ float collected_color[CHANNELS * N_ENTRIES_PER_ROUND];

	__shared__ float collected_opacity[N_ENTRIES_PER_ROUND];
	__shared__ glm::vec4 collected_g2s[4 * N_ENTRIES_PER_ROUND];


	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	[[maybe_unused]] float depth_accum = 0.f;
	[[maybe_unused]] float currentDepth = -FLT_MAX;
	[[maybe_unused]] float sortingErrorCount = 0.f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= N_ENTRIES_PER_ROUND)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int my_load_id = (block.thread_rank() / THREADS_PER_GAUSSIAN);
		int my_offset = block.thread_rank() - my_load_id * THREADS_PER_GAUSSIAN;
		int progress = i * N_ENTRIES_PER_ROUND + my_load_id;
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];

			for (int c = 0; c < CHANNELS; c += THREADS_PER_GAUSSIAN)
			{
				if (c + my_offset < CHANNELS)
					collected_color[my_load_id * CHANNELS + (c + my_offset)] = features[coll_id * CHANNELS + c + my_offset];
			}

			collected_id[my_load_id] = coll_id;

			if constexpr (EVAL_3D)
			{
				if (my_offset == 0)
					collected_opacity[my_load_id] = ((float*) conic_opacity)[coll_id];

				for (int col = 0; col < 4; col += THREADS_PER_GAUSSIAN)
					collected_g2s[my_load_id * 4 + col + my_offset] = ((glm::vec4*) gauss2screen)[coll_id * 4 + col + my_offset];
			}
			else
			{
				if (my_offset == 0)
				{
					collected_xy[my_load_id] = points_xy_image[coll_id];
					collected_conic_opacity[my_load_id] = conic_opacity[coll_id];
				}
			}			
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(N_ENTRIES_PER_ROUND, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float alpha;
			if constexpr(EVAL_3D)
			{
				float opacity = collected_opacity[j];
				glm::mat4 gauss2screen_mat;
				for (int col = 0; col < 4; col++)
					gauss2screen_mat[col] = collected_g2s[j * 4 + col];

				const glm::vec4 plane_x = gauss2screen_mat[0] - gauss2screen_mat[3] * pixf.x;
				const glm::vec4 plane_y = gauss2screen_mat[1] - gauss2screen_mat[3] * pixf.y;
				
				glm::vec4 max_pos;
				float power = -0.5f * max_contrib_ray(plane_x, plane_y, max_pos);
					
				alpha = min(0.99f, opacity * exp(power));
				if (alpha < ALPHA_THRESHOLD)
					continue;
				
				float w = glm::dot(gauss2screen_mat[3], max_pos);
				float depth = glm::dot(gauss2screen_mat[2], max_pos);
				if (depth < -w || depth > w)
					continue;
			}
			else
			{
				float4 con_o = collected_conic_opacity[j];
				float2 xy = collected_xy[j];
				float2 d = { xy.x - pixf.x, xy.y - pixf.y };
				float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;

				if (power > 0.0f)
					continue;

				alpha = min(0.99f, con_o.w * exp(power));
				if (alpha < ALPHA_THRESHOLD)
					continue;
			}

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += collected_color[j * CHANNELS + ch] * alpha * T;

			if constexpr (ENABLE_DEBUG_VIZ) {
				const glm::vec3 dir = (*cam_pos) - means3D[collected_id[j]];
				float depth = glm::length(dir);
				accumSortingErrorDepth(debugVisualizationType, currentDepth, depth, alpha, T, depth_accum, sortingErrorCount);
			}

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		if constexpr (!ENABLE_DEBUG_VIZ)
		{
			final_T[pix_id] = T;
			n_contrib[pix_id] = last_contributor;
			for (int ch = 0; ch < CHANNELS; ch++)
				out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		}
		else {
			outputDebugVis(debugVisualizationType, out_color, pix_id, contributor, T, depth_accum, sortingErrorCount, range.y - range.x, H, W);
		}
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const SplattingSettings splatting_settings,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* means3D,
	const float4* cov3D_inv,
	const float* gauss2screen,
	const float* projmatrix_inv,
	const glm::vec3* cam_pos,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	DebugVisualizationData& debugVisualization,
	float* out_color)
{

	if (splatting_settings.sort_settings.sort_mode == SortMode::GLOBAL)
	{
		#define CALL_VANILLA(ENABLE_DEBUG_VIZ, EVAL_3D) renderCUDA<NUM_CHANNELS, ENABLE_DEBUG_VIZ, EVAL_3D> <<<grid, block>>> (ranges, point_list, W, H, means2D, colors, conic_opacity, final_T, n_contrib, bg_color, debugVisualization.type, cam_pos, (glm::vec3*)means3D, gauss2screen, out_color)

		if (splatting_settings.eval_3D) {
			if (debugVisualization.type == DebugVisualization::Disabled) { CALL_VANILLA(false, true); } else { CALL_VANILLA(true, true); }
		} else {
			if (debugVisualization.type == DebugVisualization::Disabled) { CALL_VANILLA(false, false); } else { CALL_VANILLA(true, false); }
		}
		

		#undef CALL_VANILLA
	}
	else if (splatting_settings.sort_settings.sort_mode == SortMode::PER_PIXEL_KBUFFER)
	{
		#define CALL_KBUFFER_DEBUG(WINDOW, ENABLE_DEBUG_VIZ) renderkBufferCUDA<NUM_CHANNELS, WINDOW, ENABLE_DEBUG_VIZ> <<<grid, block>>> (ranges, point_list, W, H, means2D, cov3D_inv, projmatrix_inv, (float3*)cam_pos, colors, conic_opacity, final_T, n_contrib, bg_color, debugVisualization.type, out_color)
		#define CALL_KBUFFER(WINDOW) if (debugVisualization.type == DebugVisualization::Disabled) CALL_KBUFFER_DEBUG(WINDOW, false); else CALL_KBUFFER_DEBUG(WINDOW, true)
	

#ifdef STOPTHEPOP_FASTBUILD
		CALL_KBUFFER(16);
#else // STOPTHEPOP_FASTBUILD
		int window_size = splatting_settings.sort_settings.queue_sizes.per_pixel;
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
	}
	else if (splatting_settings.sort_settings.sort_mode == SortMode::PER_PIXEL_FULL)
	{
		#define CALL_FULLSORT(ENABLE_DEBUG_VIZ) renderSortedFullCUDA<NUM_CHANNELS, ENABLE_DEBUG_VIZ> <<<grid, block>>> (ranges, point_list, W, H, means2D, cov3D_inv, projmatrix_inv, (float3*) cam_pos, colors, conic_opacity, final_T, n_contrib, bg_color, debugVisualization.type, out_color)
		
		if (debugVisualization.type == DebugVisualization::Disabled) {
			CALL_FULLSORT(false);
		} else {
			CALL_FULLSORT(true);
		}

		#undef CALL_FULLSORT
	}
	else if (splatting_settings.sort_settings.sort_mode == SortMode::HIERARCHICAL)
	{
#define CALL_HIER_DEBUG(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE, EVAL_3D, DEBUG) sortGaussiansRayHierarchicalCUDA_forward<NUM_CHANNELS, HEAD_QUEUE_SIZE, MID_QUEUE_SIZE, HIER_CULLING, EVAL_3D, DEBUG><<<grid, {16, 4, 4}>>>( \
	ranges, point_list, W, H, means2D, cov3D_inv, projmatrix_inv, (float3 *)cam_pos, colors, conic_opacity, (glm::mat4*)gauss2screen, final_T, n_contrib, bg_color, debugVisualization.type, out_color)

#define CALL_HIER_EVAL3D(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE, EVAL_3D) if (debugVisualization.type == DebugVisualization::Disabled) { CALL_HIER_DEBUG(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE, EVAL_3D, false); } else { CALL_HIER_DEBUG(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE, EVAL_3D, true); }

#define CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE) if (splatting_settings.eval_3D) { CALL_HIER_EVAL3D(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE, true); } else { CALL_HIER_EVAL3D(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE, false); }

#ifdef STOPTHEPOP_FASTBUILD
#define CALL_HIER_HEAD(HIER_CULLING, MID_QUEUE_SIZE) \
	switch (splatting_settings.sort_settings.queue_sizes.per_pixel) \
	{ \
		case 4: { CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, 4); break; } \
		default: { throw std::runtime_error("Not supported head queue size"); } \
	}

#define CALL_HIER_MID(HIER_CULLING) \
	switch (splatting_settings.sort_settings.queue_sizes.tile_2x2) \
	{ \
		case 8: { CALL_HIER_HEAD(HIER_CULLING, 8); break; } \
		default: { throw std::runtime_error("Not supported mid queue size"); } \
	}
#else // STOPTHEPOP_FASTBUILD
#define CALL_HIER_HEAD(HIER_CULLING, MID_QUEUE_SIZE) \
	switch (splatting_settings.sort_settings.queue_sizes.per_pixel) \
	{ \
		case 4: { CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, 4); break; } \
		case 8: { CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, 8); break; } \
		case 16: { CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, 16); break; } \
		default: { throw std::runtime_error("Not supported head queue size"); } \
	}

#define CALL_HIER_MID(HIER_CULLING) \
	switch (splatting_settings.sort_settings.queue_sizes.tile_2x2) \
	{ \
		case 8: { CALL_HIER_HEAD(HIER_CULLING, 8); break; } \
		case 12: { CALL_HIER_HEAD(HIER_CULLING, 12); break; } \
		case 20: { CALL_HIER_HEAD(HIER_CULLING, 20); break; } \
		default: { throw std::runtime_error("Not supported mid queue size"); } \
	}
#endif // STOPTHEPOP_FASTBUILD

	if (splatting_settings.culling_settings.hierarchical_4x4_culling) {
		CALL_HIER_MID(true);
	} else {
		CALL_HIER_MID(false);
	}

#undef CALL_HIER_MID
#undef CALL_HIER_HEAD
#undef CALL_HIER
#undef CALL_HIER_EVAL3D
#undef CALL_HIER_DEBUG
	}
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* rects,
	const SplattingSettings splatting_settings,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float4* cov3D_invs,
	float* gauss2screen,
	float* filter3D,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
#define PREPROCESS_CALL(TBC, LB, EVAL_3D) \
	preprocessCUDA<NUM_CHANNELS, TBC, LB, EVAL_3D> << <(P + 255) / 256, 256 >> > ( \
		P, D, M, \
		means3D, \
		scales, \
		scale_modifier, \
		rotations, \
		opacities, \
		shs, \
		clamped, \
		cov3D_precomp, \
		colors_precomp, \
		viewmatrix,  \
		projmatrix, \
		cam_pos, \
		W, H, \
		tan_fovx, tan_fovy, \
		focal_x, focal_y, \
		radii, \
		rects, \
		splatting_settings.sort_settings.sort_order, \
		splatting_settings.culling_settings.rect_bounding, \
		splatting_settings.culling_settings.tight_opacity_bounding, \
		splatting_settings.proper_ewa_scaling, \
		splatting_settings.new_aabb, \
		splatting_settings.near_clipping, \
		means2D, \
		depths, \
		cov3Ds, \
		cov3D_invs, \
		gauss2screen, \
		filter3D, \
		rgb, \
		conic_opacity, \
		grid, \
		tiles_touched, \
		prefiltered \
		);

#define PREPROCESS_EVAL_3D(TBC, LB) if (splatting_settings.eval_3D) { PREPROCESS_CALL(TBC, LB, true); } else { PREPROCESS_CALL(TBC, LB, false); }

	if (splatting_settings.culling_settings.tile_based_culling)
	{
		if (splatting_settings.load_balancing) {
			PREPROCESS_EVAL_3D(true, true);
		} else {
			PREPROCESS_EVAL_3D(true, false);
		}
	}
	else
	{
		if (splatting_settings.load_balancing) {
			PREPROCESS_EVAL_3D(false, true);
		} else {
			PREPROCESS_EVAL_3D(false, false);
		}
	}

#undef PREPROCESS_CALL
#undef PREPROCESS_EVAL_3D
}

void FORWARD::duplicate(int P,
						const float2 *means2D,
						const float4 *conic_opacity,
						const float *gauss2screen,
						const int *radii,
						const float2 *rects2D,
						const uint32_t *offsets,
						const float *depths,
						const float4 *cov3D_invs,
						const SplattingSettings splatting_settings,
						const float *projmatrix,
						const float *inv_viewprojmatrix,
						const float *cam_pos,
						const int W, int H,
						uint64_t *gaussian_keys_unsorted,
						uint32_t *gaussian_values_unsorted,
						dim3 grid)
{
	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	#define CALL_DUPLICATE_EXTENDED(TBC, LB, SORT_ORDER, EVAL_3D) \
		duplicateWithKeys_extended<TBC, LB, SORT_ORDER, EVAL_3D> << <(P + 255) / 256, 256 >> > ( \
				P, \
				means2D, \
				depths, \
				cov3D_invs, \
				conic_opacity, \
				(glm::mat4*) gauss2screen, \
				projmatrix, \
				inv_viewprojmatrix, \
				(glm::vec3*) cam_pos, \
				W, H, \
				offsets, \
				gaussian_keys_unsorted, \
				gaussian_values_unsorted, \
				radii, \
				rects2D, \
				grid)

	#define CALL_DUPLICATE_EVAL3D(TBC, LB, SORT_ORDER) if(splatting_settings.eval_3D) { CALL_DUPLICATE_EXTENDED(TBC, LB, SORT_ORDER, true); } else { CALL_DUPLICATE_EXTENDED(TBC, LB, SORT_ORDER, false); }

	#define CALL_DUPLICATE_SORT_ORDER(SORT_ORDER) \
		if (splatting_settings.culling_settings.tile_based_culling) \
		{ \
			if (splatting_settings.load_balancing) { \
				CALL_DUPLICATE_EVAL3D(true, true, SORT_ORDER) \
			} else { \
				CALL_DUPLICATE_EVAL3D(true, false, SORT_ORDER) \
			} \
		} else { \
			if (splatting_settings.load_balancing) { \
				CALL_DUPLICATE_EVAL3D(false, true, SORT_ORDER) \
			} else { \
				CALL_DUPLICATE_EVAL3D(false, false, SORT_ORDER) \
			} \
		}

		switch (splatting_settings.sort_settings.sort_order)
		{
			case GlobalSortOrder::VIEWSPACE_Z:
			case GlobalSortOrder::DISTANCE:
			{
				if (!splatting_settings.load_balancing && !splatting_settings.culling_settings.tile_based_culling)
				{
					duplicateWithKeysCUDA<<<(P + 255) / 256, 256>>>(
						P,
						rects2D,
						means2D,
						depths,
						offsets,
						gaussian_keys_unsorted,
						gaussian_values_unsorted,
						radii,
						grid);
				}
				else
				{
					// viewspace-z and distance treated equally
					CALL_DUPLICATE_SORT_ORDER(GlobalSortOrder::VIEWSPACE_Z)
				}
				break;
			}
			case GlobalSortOrder::PER_TILE_DEPTH_CENTER:
			{
				CALL_DUPLICATE_SORT_ORDER(GlobalSortOrder::PER_TILE_DEPTH_CENTER)
				break;
			}
			case GlobalSortOrder::PER_TILE_DEPTH_MAXPOS:
			{
				CALL_DUPLICATE_SORT_ORDER(GlobalSortOrder::PER_TILE_DEPTH_MAXPOS)
				break;
			}
		}
	#undef CALL_DUPLICATE_EXTENDED
	#undef CALL_DUPLICATE_EVAL3D
	#undef CALL_DUPLICATE_SORT_ORDER
}

template<uint32_t CHANNELS, bool DEPTH>
__global__ void render_debug_CUDA(int P, const float* __restrict__ min_max_contrib, float* out_color, bool debug_normalize, float debug_norm_min, float debug_norm_max)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float min = min_max_contrib[0];
	float max = min_max_contrib[1];

	if (debug_normalize)
	{
		min = debug_norm_min;
		max = debug_norm_max;
	}

	float alpha = 0.f;
	if constexpr (DEPTH) 
	{
		float T = (out_color + P)[idx];
		alpha = fminf(fmaxf(out_color[idx] + T * max, min), max) / static_cast<float>(max - min);
	}
	else
	{
		alpha = fminf(fmaxf(out_color[idx], min), max) / static_cast<float>(max - min);
	}

	glm::vec3 output;
	if constexpr (DEPTH)
		{
			output = colormapTurbo(alpha);
		}
		else 
		{
			output = colormapMagma(alpha);
		}
	for (int ch = 0; ch < CHANNELS; ch++)
	{
		out_color[ch * P + idx] = output[ch];
	}
}

void FORWARD::render_debug(DebugVisualizationData& debugVisualization, int P, float* out_color, float* min_max_contrib)
{
	#define CALL_DEBUG(DEPTH) render_debug_CUDA<NUM_CHANNELS, DEPTH><<<(P + 255) / 256, 256>>>(P, min_max_contrib, out_color, debugVisualization.debug_normalize, debugVisualization.minMax[0], debugVisualization.minMax[1])
	
	if (sortQualityDebug::isMagma(debugVisualization.type))
	{
		CALL_DEBUG(false);
	}
	else{
		CALL_DEBUG(true);
	}

	#undef CALL_DEBUG
}