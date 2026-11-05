/*
 * Python binding for the official GaussianHierarchyCreator pipeline.
 *
 * The body of `create` is the `main()` of vendor/gaussian_hierarchy/mainHierarchyCreator.cpp
 * (github.com/graphdeco-inria/gaussian-hierarchy @677c855, Copyright (C) 2024 Inria,
 * non-commercial research licence, see vendor/LICENSE_hierarchical-3d-gaussians.md), turned into
 * a function so the plugin can call it in-process. No step, constant or order is changed:
 * load .ply (sigmoid/exp activations in Loader::loadPly), drop Inf/NaN/zero-opacity Gaussians,
 * PointbasedKdTreeGenerator (BVH median split), ClusterMerger (Eq. merge), RotationAligner
 * (orientation matching), AppearanceFilter (compaction, limit 0.0005, multiplier 2), anchors,
 * Writer::writeHierarchy(compressed = true). Scaffold/skybox arguments are not exposed
 * (skyboxpoints = 0, as when the official binary is called without a scaffold directory).
 */
#include <torch/extension.h>
#include <cmath>
#include <string>
#include <vector>
#include "loader.h"
#include "writer.h"
#include "PointbasedKdTreeGenerator.h"
#include "ClusterMerger.h"
#include "common.h"
#include "appearance_filter.h"
#include "rotation_aligner.h"

int create(std::string plyfile, std::string sourcedir, std::string outdir)
{
	int skyboxpoints = 0;
	std::vector<Gaussian> gaussians_unfiltered;
	Loader::loadPly(plyfile.c_str(), gaussians_unfiltered, skyboxpoints);

	int valid = 0;
	std::vector<Gaussian> gaussians(gaussians_unfiltered.size());
	for (int i = 0; i < gaussians_unfiltered.size(); i++)
	{
		Gaussian& g = gaussians_unfiltered[i];
		if (std::isinf(g.opacity) || std::isnan(g.opacity) || g.scale.hasNaN() ||
			g.rotation.hasNaN() || g.position.hasNaN() || g.shs.hasNaN() || g.opacity == 0)
			continue;
		gaussians[valid] = g;
		valid++;
	}
	gaussians.resize(valid);

	PointbasedKdTreeGenerator generator;
	auto root = generator.generate(gaussians);

	ClusterMerger merger;
	merger.merge(root, gaussians);

	RotationAligner::align(root, gaussians);

	float limit = 0.0005f;
	AppearanceFilter filter;
	filter.init(sourcedir.c_str());
	filter.filter(root, gaussians, limit, 2.0f);
	filter.writeAnchors((outdir + "/anchors.bin").c_str(), root, gaussians, limit);

	Writer::writeHierarchy((outdir + "/hierarchy.hier").c_str(), gaussians, root, true);
	return valid;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("create", &create, "Official GaussianHierarchyCreator pipeline (returns #valid leaves)");
}
