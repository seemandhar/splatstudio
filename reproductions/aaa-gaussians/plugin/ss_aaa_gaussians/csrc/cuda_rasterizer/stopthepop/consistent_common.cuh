/*
 * Copyright (C) 2025, Graz University of Technology
 * This code is licensed under the MIT license (see LICENSE.txt in this folder for details)
 */

 #pragma once

 #include "../auxiliary.h"
 
 #include <math_constants.h>
 
 __forceinline__ __device__ glm::mat4 compute_gauss2screen(
     const glm::vec3 mean3D,
     const glm::vec3 scale,
     const glm::vec4 rot,
     const float scale_modifier,
 
     const glm::vec3 campos_world,
     const glm::mat4 world2screen,
     const glm::mat4 view_mat,
     const float focal,
     const float kernel_size,
 
     glm::mat4& gauss2view,
     glm::vec3& campos_gauss,
     float &dilation_factor,
     float filter_3d = 0.0f)
 {
     glm::mat3 R = quat2mat(rot);
     
     glm::vec3 scale_dilated = scale;
     if (kernel_size > 0.0f)
     {
         glm::vec3 viewray_world = glm::normalize(mean3D - campos_world);
         glm::vec3 viewray_gauss = R * viewray_world;
         glm::vec3 r = viewray_gauss * viewray_gauss;
 
         glm::vec4 mean3D_view = view_mat * glm::vec4(mean3D, 1.0f);
         float scale_mip_filter = sq(mean3D_view.z / focal) * kernel_size;

         scale_mip_filter = fmaxf(sq(filter_3d), scale_mip_filter);
 
         scale_dilated = glm::sqrt(scale * scale + scale_mip_filter);
 
         glm::vec3 s = scale * scale;
         glm::vec3 s_dil = scale_dilated * scale_dilated;
 
         float det_mul_ray_var = glm::dot(r, glm::vec3(s.y * s.z, s.z * s.x, s.x * s.y));
         float det_mul_ray_var_dil = glm::dot(r, glm::vec3(s_dil.y * s_dil.z, s_dil.z * s_dil.x, s_dil.x * s_dil.y));
         dilation_factor = sqrt(det_mul_ray_var / det_mul_ray_var_dil);
     }
 
     glm::mat3 S = glm::mat3(1.0f);
     S[0][0] = scale_dilated.x;
     S[1][1] = scale_dilated.y;
     S[2][2] = scale_dilated.z;
     S *= sqrt(scale_modifier);
 
     glm::mat3 L = glm::transpose(S * R);
 
     // center of Gaussians in the camera coordinate
     glm::mat4 gauss2world(glm::vec4(L[0], 0.0f),
                           glm::vec4(L[1], 0.0f),
                           glm::vec4(L[2], 0.0f),
                           glm::vec4(mean3D.x, mean3D.y, mean3D.z, 1.0f));
     
     glm::mat3 S_inv = glm::mat3(1.0f);
     S_inv[0][0] = 1.0f / scale_dilated.x;
     S_inv[1][1] = 1.0f / scale_dilated.y;
     S_inv[2][2] = 1.0f / scale_dilated.z;
     S_inv /= sqrt(scale_modifier);
     glm::mat3 world2gauss = S_inv * R;
 
     campos_gauss = world2gauss * (campos_world - mean3D);
 
     gauss2view = glm::transpose(view_mat * gauss2world);
     return glm::transpose(world2screen * gauss2world);
 }

 __device__ inline float max_contrib_plane(const glm::vec4 plane, const glm::mat4 g2s, glm::vec4& max_pos_screen)
 {
    float norm = plane.w / (plane.x * plane.x + plane.y * plane.y + plane.z * plane.z);
    
    glm::vec4 max_pos_gauss = -plane * norm;
    max_pos_screen = glm::transpose(g2s) * glm::vec4(glm::vec3(max_pos_gauss), 1.0f);
    return -max_pos_gauss.w;
 }
 
 __device__ inline float max_contrib_ray(const glm::vec4 plane_a, const glm::vec4 plane_b, glm::vec4& max_pos)
 {
     glm::vec3 d = {
         plane_a.y * plane_b.z - plane_a.z * plane_b.y,
         plane_a.z * plane_b.x - plane_a.x * plane_b.z,
         plane_a.x * plane_b.y - plane_a.y * plane_b.x,
     };
 
     glm::vec3 m = {
         plane_a.w * plane_b.x - plane_a.x * plane_b.w,
         plane_a.w * plane_b.y - plane_a.y * plane_b.w,
         plane_a.w * plane_b.z - plane_a.z * plane_b.w,
     };
 
     glm::vec3 m_div_dd = m / glm::dot(d, d);
     max_pos = glm::vec4(glm::cross(d, m_div_dd), 1.0f);
     return glm::dot(m, m_div_dd);
 }
 
 __device__ inline float max_contrib_ray(const glm::vec4 plane_a, const glm::vec4 plane_b, const glm::mat4 g2s, glm::vec4& max_pos_screen)
 {
    glm::vec4 max_pos_gauss;
    float contrib = max_contrib_ray(plane_a, plane_b, max_pos_gauss);
    max_pos_screen = glm::transpose(g2s) * max_pos_gauss;
    return contrib;
 }

template<int DIM>
__device__ inline bool in_screen_range(const glm::vec4 pos_screen, glm::vec3 range_from, glm::vec3 extent, float& d)
{
    d = pos_screen[DIM] - range_from[DIM] * pos_screen.w;
    return d > 0.0f && d < extent[DIM] * pos_screen.w;
}

template<int DIM>
__device__ inline bool in_screen_range(const glm::vec4 pos_screen, glm::vec3 range_from, glm::vec3 extent)
{
    float d;
    return in_screen_range<DIM>(pos_screen, range_from, extent, d);
}
 
template<bool KNOWN_DIMENSIONS = false, uint32_t PATCH_WIDTH = 0, uint32_t PATCH_HEIGHT = 0>
__device__ inline float max_contrib_gaussian_frustum_3D(
     const glm::vec2 rect_min,
     const glm::vec2 rect_max,
     const glm::mat4 g2s,
     float& max_pos_depth)
{
    float max_contrib = FLT_MAX;
    glm::vec4 max_pos_screen = glm::vec4(1.0f);

    const glm::vec3 RANGE_FROM(rect_min.x, rect_min.y, Z_NEAR);
    const glm::vec3 EXTENT = {
        KNOWN_DIMENSIONS ? PATCH_WIDTH : (rect_max.x - rect_min.x),
        KNOWN_DIMENSIONS ? PATCH_HEIGHT : (rect_max.y - rect_min.y),
        Z_FAR - Z_NEAR
    };

    glm::vec4 mean_screen(g2s[0][3], g2s[1][3], g2s[2][3], g2s[3][3]);

    glm::vec3 d;
    bool between_x = in_screen_range<0>(mean_screen, RANGE_FROM, EXTENT, d.x);
    bool between_y = in_screen_range<1>(mean_screen, RANGE_FROM, EXTENT, d.y);
    bool between_z = in_screen_range<2>(mean_screen, RANGE_FROM, EXTENT, d.z);
    if (between_x && between_y && between_z)
    {
        max_pos_screen = mean_screen;
        max_contrib = 0.0f;
    }
    else
    {
        float dx = copysignf(EXTENT.x * 0.5f, d.x - (EXTENT.x * 0.5f * mean_screen.w));
        float dy = copysignf(EXTENT.y * 0.5f, d.y - (EXTENT.y * 0.5f * mean_screen.w));

        glm::vec4 closer_plane_x = g2s[0] - g2s[3] * (RANGE_FROM.x + EXTENT.x * 0.5f + dx);
        glm::vec4 closer_plane_y = g2s[1] - g2s[3] * (RANGE_FROM.y + EXTENT.y * 0.5f + dy);

        glm::vec4 pos_screen;
        float contrib = max_contrib_plane(closer_plane_x, g2s, pos_screen);
        if (contrib < max_contrib && in_screen_range<1>(pos_screen, RANGE_FROM, EXTENT) && in_screen_range<2>(pos_screen, RANGE_FROM, EXTENT))
        {
            max_contrib = contrib;
            max_pos_screen = pos_screen;
        }

        contrib = max_contrib_plane(closer_plane_y, g2s, pos_screen);
        if (contrib < max_contrib && in_screen_range<0>(pos_screen, RANGE_FROM, EXTENT) && in_screen_range<2>(pos_screen, RANGE_FROM, EXTENT))
        {
            max_contrib = contrib;
            max_pos_screen = pos_screen;
        }

        contrib = max_contrib_ray(closer_plane_x, closer_plane_y, g2s, pos_screen);
        if (contrib < max_contrib && in_screen_range<2>(pos_screen, RANGE_FROM, EXTENT))
        {
            max_contrib = contrib;
            max_pos_screen = pos_screen;
        }

        glm::vec4 other_plane_y = g2s[1] - g2s[3] * (RANGE_FROM.y + EXTENT.y * 0.5f - dy);
        contrib = max_contrib_ray(closer_plane_x, other_plane_y, g2s, pos_screen);
        if (contrib < max_contrib && in_screen_range<2>(pos_screen, RANGE_FROM, EXTENT))
        {
            max_contrib = contrib;
            max_pos_screen = pos_screen;
        }

        glm::vec4 other_plane_x = g2s[0] - g2s[3] * (RANGE_FROM.x + EXTENT.x * 0.5f - dx);
        contrib = max_contrib_ray(other_plane_x, closer_plane_y, g2s, pos_screen);
        if (contrib < max_contrib && in_screen_range<2>(pos_screen, RANGE_FROM, EXTENT))
        {
            max_contrib = contrib;
            max_pos_screen = pos_screen;
        }
    }

    max_pos_depth = max_pos_screen.z / max_pos_screen.w;
    return 0.5f * max_contrib;
 }

 __forceinline__ __device__ float normalize_angle(float theta)
 {
    theta = fmod(theta, 2 * CUDART_PI_F); // Wrap to (-2π, 2π)
    if (theta > CUDART_PI_F) theta -= 2 * CUDART_PI_F; // Adjust to (-π, π]
    if (theta <= -CUDART_PI_F) theta += 2 * CUDART_PI_F;
    return theta;
 }
 
__forceinline__ __device__ bool compute_aabb_view(
    const glm::mat4 gauss2view,
    const glm::vec3 mean3D_view,
    const glm::vec3 up,
    float focal_x, float focal_y,
    float W, float H,
    float cutoff,
    glm::vec2& mean2D,
    glm::vec2& extent2D)
{
    glm::vec4 t(cutoff, cutoff, cutoff, -1);
    glm::vec3 viewdir = glm::normalize(mean3D_view);

    auto compute_theta = [&](int axis)
    { 
        float theta_mu = atan2((axis == 0 ? viewdir.x : viewdir.y), viewdir.z);

        const glm::mat4& T = gauss2view;
        float squared_axis = glm::dot(t, T[axis] * T[axis]);
        float squared_z = glm::dot(t, T[2] * T[2]);
        float mid = glm::dot(t, T[axis] * T[2]);

        float inside_sqrt = sq(mid) - squared_z * squared_axis;

        // set default to full screen
        glm::vec2 result(-(CUDART_PIO2_F - 1e-5f), (CUDART_PIO2_F - 1e-5f));
        if (inside_sqrt > 0.0f)
        {
            float theta_mid = atan2(-mid, -squared_z);
            glm::vec2 theta_axis_orig(
                atan2(-(mid + sqrt(inside_sqrt)), -squared_z),
                atan2(-(mid - sqrt(inside_sqrt)), -squared_z)
            );
            glm::vec2 theta_axis = theta_axis_orig;
            
            // rotate to get correct order
            while (theta_axis[0] > theta_mu)                   theta_axis[0] -= CUDART_PI_F;
            while (theta_axis[0] < (theta_mu - CUDART_PI_F))   theta_axis[0] += CUDART_PI_F;

            while (theta_axis[1] < theta_mu)                   theta_axis[1] += CUDART_PI_F;
            while (theta_axis[1] > (theta_mu + CUDART_PI_F))   theta_axis[1] -= CUDART_PI_F;

            glm::vec2 theta_axis_norm(normalize_angle(theta_axis[0]), normalize_angle(theta_axis[1]));
            if (theta_mu < 0.0f && fabs(theta_axis_norm[0]) < fabs(theta_axis_norm[1]))
            {
                theta_axis[0] += 2.0f * CUDART_PI_F;
                theta_axis[1] += 2.0f * CUDART_PI_F;
            }
            else if (theta_mu > 0.0f && fabs(theta_axis_norm[1]) < fabs(theta_axis_norm[0]))
            {
                theta_axis[0] -= 2.0f * CUDART_PI_F;
                theta_axis[1] -= 2.0f * CUDART_PI_F;
            }

            // constrain angles to be in front of the camera
            result = glm::vec2(fmaxf(result[0], theta_axis[0]), fminf(result[1], theta_axis[1]));
        }

        return glm::tan(result);
    };

    glm::vec2 bounds_x = W/2.0f + focal_x * compute_theta(0);
    glm::vec2 bounds_y = H/2.0f + focal_y * compute_theta(1);

    mean2D = {(bounds_x[1] + bounds_x[0]) / 2.0f, (bounds_y[1] + bounds_y[0]) / 2.0f};
    extent2D = {(bounds_x[1] - bounds_x[0]) / 2.0f, (bounds_y[1] - bounds_y[0]) / 2.0f};
    return true;
}

__forceinline__ __device__ bool compute_aabb_screen(
    const glm::mat4 T,
    const glm::vec3 mean3D_view,
    const glm::vec3 up,
    float focal_x, float focal_y,
    float W, float H,
    float cutoff,
    glm::vec2& mean2D,
    glm::vec2& extent2D)
{
	// Screen-Space Bounding as proposed by Hahlbohm et al. in
	// "Efficient Perspective-Correct 3D Gaussian Splatting Using Hybrid Transparency"

	glm::vec4 t = glm::vec4(cutoff, cutoff, cutoff, -1.0f);
	float s = glm::dot(t, T[3] * T[3]);
	if (s >= 0.0f) return false;

    glm::vec4 f = (1.0f / s) * t;

	glm::vec3 p = glm::vec3(
		glm::dot(f, T[0] * T[3]),
		glm::dot(f, T[1] * T[3]),
		glm::dot(f, T[2] * T[3])
	);
	glm::vec3 h_inside_sqrt = p * p - glm::vec3(
		glm::dot(f, T[0] * T[0]),
		glm::dot(f, T[1] * T[1]),
		glm::dot(f, T[2] * T[2])
	);

    float extent_z = sqrtf(fmaxf(h_inside_sqrt.z, 0.0f));
    glm::vec2 z_bounds(p.z - extent_z, p.z + extent_z);
    if (z_bounds.x < -1.0f || z_bounds.y > 1.0f) return false;

	mean2D = { p.x, p.y };
	extent2D = { sqrtf(fmaxf(h_inside_sqrt.x, 0.0f)), sqrtf(fmaxf(h_inside_sqrt.y, 0.0f))};
	return true;
}
