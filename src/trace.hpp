#pragma once

#include "light.hpp"
#include "jobs.hpp"
#include "xoshiro.hpp"
#include "sdf.hpp"

inline
std::pair<bool, float> rouletteWeight(float rnd1, const float stopProb) {
	if (rnd1 <= stopProb) { return std::make_pair(true, 1.0); }
	return std::make_pair(false, 1.0 / (1.0 - stopProb));
}

light::Vector trace(const light::Ray& cameraRay, const light::RayTracerContext& tracer, TraceTileJob& job) {
	using namespace light;
	static const Vector zero(0, 0, 0);
	static const Vector one(1, 1, 1);
	std::vector<Contribution> contributions;
	contributions.reserve(2*tracer.rouletteDepth);
	if (job.pathCapture) {
		job.vertices.clear();
		job.vertices.reserve(contributions.capacity());
	}
	bool hitEmitter = false;
	auto gen = job.getGenerators();

	std::uint32_t depth = 0;

	// Loop to trace the ray through the scence and produce the ray path:
	auto ray = cameraRay;
	while (true) {
		// Russian roulette ray termination:
		float rrFactor = 1.0;
		if (depth >= tracer.rouletteDepth) {
			bool stop;
			const float rnd1 = xoshiro::uniform_0_1(gen.rng);
			std::tie(stop, rrFactor) = rouletteWeight(rnd1, tracer.stopProb);
			if (stop) { break; }
		}

		Intersection intersection = tracer.scene.intersect(ray);
		if (!intersection) { break; }

		// Compute hit point and surface normal there:
		ray.origin += ray.direction * intersection.t;
		Vector normal = intersection.object->normal(ray.origin);

		if (job.pathCapture) {
			job.vertices.push_back(ray.origin);
		}

		if (intersection.object->emissive) {
			contributions.push_back({intersection.object->emission, rrFactor, Contribution::Type::EMIT});
			hitEmitter = true;
		}

		if (intersection.object->type == Material::diffuse) {
			const auto rnd1 = xoshiro::uniform_0_1(gen.rng);
			const auto rnd2 = xoshiro::uniform_0_1(gen.rng);
			const auto result = diffuse(ray, normal, intersection, rrFactor, rnd1, rnd2);
			contributions.push_back(result);
		} else if (intersection.object->type == Material::specular) {
			reflect(ray, normal);
			contributions.push_back({zero, rrFactor, Contribution::Type::SPECULAR});
		} else if (intersection.object->type == Material::refractive) {
			const auto rnd1 = xoshiro::uniform_0_1(gen.rng);
			refract(ray, normal, tracer, rnd1);
			contributions.push_back({zero, 1.15f * rrFactor, Contribution::Type::REFLECT});
		}

		depth += 1;
	}

	job.totalRayCasts += contributions.size();
	job.maxPathLength = std::max(job.maxPathLength, contributions.size());
	job.nonZeroContribution = hitEmitter;

	// Combine all the material contributions along the ray path:
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
