#pragma once

#include "light.hpp"
#include "jobs.hpp"
#include "xoshiro.hpp"

inline
std::pair<bool, float> rouletteWeight(xoshiro::State& state, const float stopProb) {
	if (xoshiro::rnd2(state) <= stopProb) { return std::make_pair(true, 1.0); }
	return std::make_pair(false, 1.0 / (1.0 - stopProb));
}

light::Vector trace(light::Ray& ray, const light::RayTracerContext& tracer, TraceTileJob& job);

// Diffuse BRDF - choose an outgoing direction with hemisphere sampling.
light::Contribution diffuse(light::Ray& ray, light::Vector normal,
										 const light::Intersection& intersection, float rrFactor,
										 Generators gen) {
	using namespace light;
	Vector rotX, rotY;
	std::tie(rotX, rotY, std::ignore) = orthonormalSystem(normal);

#ifdef USE_EIGEN
	Eigen::Matrix3f R;
	R << rotX, rotY, N;
	ray.direction = R * hemisphere(xoshiro::rnd2(rngState), xoshiro::rnd2(rngState));	// Rotation applied to normalised vector is still unit.
#else
	const auto sampledDir = hemisphere(xoshiro::rnd2(gen.rng), xoshiro::rnd2(gen.rng));
	ray.direction = light::Vector(
		Vector(rotX.x, rotY.x, normal.x).dot(sampledDir),
		Vector(rotX.y, rotY.y, normal.y).dot(sampledDir),
		Vector(rotX.z, rotY.z, normal.z).dot(sampledDir)
	);
#endif

	float weight = ray.direction.dot(normal) * .1f * rrFactor;
	return Contribution{intersection.object->colour, weight, Contribution::Type::DIFFUSE};
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
						 const light::RayTracerContext& tracer, xoshiro::State& state) {
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
	if (cost2 > 0 && xoshiro::rnd2(state) > Rprob) { // refraction direction
		ray.direction = ((ray.direction*n)+(normal*(n*cost1-sqrt(cost2)))).normalized();
	} else { // reflection direction
		ray.direction = (ray.direction+normal*(cost1*2)).normalized();
	}
}

light::Vector trace(light::Ray& ray, const light::RayTracerContext& tracer, TraceTileJob& job) {
	using namespace light;
	static const Vector zero(0, 0, 0);
	static const Vector one(1, 1, 1);
	std::vector<Contribution> contributions;
	contributions.reserve(2*tracer.rouletteDepth);
	bool hitEmitter = false;
	auto gen = job.getGenerators();

	std::uint32_t depth = 0;

	while (true) {
		// Russian roulette ray termination:
		float rrFactor = 1.0;
		if (depth >= tracer.rouletteDepth) {
			bool stop;
			std::tie(stop, rrFactor) = rouletteWeight(gen.rng, tracer.stopProb);
			if (stop) { break; }
		}

		Intersection intersection = tracer.scene.intersect(ray);
		if (!intersection) { break; }

		// Travel the ray to the hit point where the closest object lies and compute the surface normal there.
		ray.origin += ray.direction * intersection.t;
		Vector normal = intersection.object->normal(ray.origin);

		// Add the emission, the L_e(x,w) part of the rendering equation, but scale it with the Russian Roulette probability weight.
		if (intersection.object->emissive) {
			contributions.push_back({intersection.object->emission, rrFactor, Contribution::Type::EMIT});
			hitEmitter = true;
		}

		if (intersection.object->type == Material::diffuse) {
			const auto result = diffuse(ray, normal, intersection, rrFactor, gen);
			contributions.push_back(result);
		} else if (intersection.object->type == Material::specular) {
			reflect(ray, normal);
			contributions.push_back({zero, rrFactor, Contribution::Type::SPECULAR});
		} else if (intersection.object->type == Material::refractive) {
			refract(ray, normal, tracer, gen.rng);
			contributions.push_back({zero, 1.15f * rrFactor, Contribution::Type::REFLECT});
		}

		depth += 1;
	}

	job.totalRayCasts += contributions.size();
	job.maxPathLength = std::max(job.maxPathLength, contributions.size());

	Vector total = zero;
	if (hitEmitter) {
		while (!contributions.empty()) {
			auto c = contributions.back();
			contributions.pop_back();

			switch (c.type) {
			case Contribution::Type::DIFFUSE:
				total = total.cwiseProduct(c.clr) * c.weight;
				break;
			case Contribution::Type::EMIT:
				total += c.clr * c.weight;
				break;
			case Contribution::Type::SPECULAR:
				total *= c.weight;
				break;
			case Contribution::Type::REFLECT:
				total *= c.weight;
				break;
			case Contribution::Type::SKIP:
			default:
				break;
			}
		}
	}
	return total;
}