#pragma once

#include "light.hpp"

// Diffuse BRDF - choose an outgoing direction with hemisphere sampling.
light::Contribution diffuse(light::Ray& ray, light::Vector normal,
										 const light::Intersection& intersection, float rrFactor,
										 float rnd1, float rnd2) {
	using namespace light;
	Vector rotX, rotY;
	std::tie(rotX, rotY, std::ignore) = orthonormalSystem(normal);

#ifdef USE_EIGEN
	Eigen::Matrix3f R;
	R << rotX, rotY, N;
	ray.direction = R * hemisphere(rnd1, rnd2);	// Rotation applied to normalised vector is still unit.
#else
	const auto sampledDir = hemisphere(rnd1, rnd2);
	ray.direction = light::Vector(
		Vector(rotX.x, rotY.x, normal.x).dot(sampledDir),
		Vector(rotX.y, rotY.y, normal.y).dot(sampledDir),
		Vector(rotX.z, rotY.z, normal.z).dot(sampledDir)
	);
#endif

	float weight = ray.direction.dot(normal) * .1f * rrFactor;
	return Contribution{intersection.material->colour, weight, Contribution::Type::DIFFUSE};
}

// Specular BRDF - this is a singularity in the rendering equation that follows
// delta distribution, therefore we handle this case explicitly - one incoming
// direction -> one outgoing direction, that is, the perfect reflection direction.
void reflect(light::Ray& ray, light::Vector normal) {
	auto cost = ray.direction.dot(normal);
	ray.direction = (ray.direction - normal * (cost * 2.f)).normalized();
}

// Glass/refractive BRDF - we use the vector version of Snell's law and Fresnel's law
// to compute the outgoing reflection and refraction directions and probability weights.
void refract(light::Ray& ray, light::Vector normal,
						 const light::RayTracerContext& tracer, float rnd1) {
	auto n = tracer.refractiveIndex;
	auto R0 = (1.0-n)/(1.0+n);
	R0 = R0*R0;
	if(normal.dot(ray.direction) > 0) { // we're inside the medium
		normal = -normal;
	} else {
		n = 1 / n;
	}
	auto cost1 = -normal.dot(ray.direction); // cosine of theta_1
	auto cost2 = 1.0 - n*n*(1.0-cost1*cost1); // cosine of theta_2
	auto Rprob = R0 + (1.0-R0) * powf(1.0 - cost1, 5.0); // Schlick-approximation
	if (cost2 > 0 && rnd1 > Rprob) { // refraction direction
		ray.direction = ((ray.direction*n)+(normal*(n*cost1-sqrt(cost2)))).normalized();
	} else { // reflection direction
		ray.direction = (ray.direction+normal*(cost1*2)).normalized();
	}
}
