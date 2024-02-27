#pragma once
#include "SimpleMesh.h"

class ProcrustesAligner {
public:
	Matrix4f estimatePose(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints) {
		ASSERT(sourcePoints.size() == targetPoints.size() && "The number of source and target points should be the same, since every source point is matched with corresponding target point.");

		// We estimate the pose between source and target points using Procrustes algorithm.
		// Our shapes have the same scale, therefore we don't estimate scale. We estimated rotation and translation
		// from source points to target points.

		auto sourceMean = computeMean(sourcePoints);
		auto targetMean = computeMean(targetPoints);
		
		Matrix3f rotation = estimateRotation(sourcePoints, sourceMean, targetPoints, targetMean);
		Vector3f translation = computeTranslation(sourceMean, targetMean, rotation);

		Matrix4f estimatedPose = Matrix4f::Identity();
		
		estimatedPose.block(0, 0, 3, 3) = rotation;
		estimatedPose.block(0, 3, 3, 1) = -rotation * sourceMean + translation + sourceMean; // Movement

		return estimatedPose;
	}

private:
	Vector3f computeMean(const std::vector<Vector3f>& points) {
		size_t size = points.size();
		Vector3f mean = Vector3f::Zero();
		for (auto p : points)
			mean += p;

		return mean / size;
	}

	Matrix3f estimateRotation(const std::vector<Vector3f>& sourcePoints, const Vector3f& sourceMean, const std::vector<Vector3f>& targetPoints, const Vector3f& targetMean) {
		Matrix3f rotation = Matrix3f::Identity(); 
		Matrix3f Pippo = Matrix3f::Zero();

		for (size_t idx = 0; idx < sourcePoints.size(); ++idx) {
			Pippo += (targetPoints[idx] - targetMean) * (sourcePoints[idx] - sourceMean).transpose();
		}
		Eigen::JacobiSVD<Matrix3f> svd(Pippo, Eigen::ComputeFullU | Eigen::ComputeFullV);

		rotation = svd.matrixU() * svd.matrixV().transpose();

        return rotation;
	}

	Vector3f computeTranslation(const Vector3f& sourceMean, const Vector3f& targetMean, const Matrix3f& rotation) {
		Vector3f translation = Vector3f::Zero();
		translation = targetMean - sourceMean;
		
        return translation;
	}
};
