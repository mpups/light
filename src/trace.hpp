#pragma once

#include "light.hpp"
#include "jobs.hpp"
#include "xoshiro.hpp"
#include "sdf.hpp"

template <typename SceneType>
light::Vector trace(const light::Ray& cameraRay,
                    const light::RayTracerContext<SceneType>& tracer,
                    TraceTileJob& job) {
  using namespace light;
  static const Vector zero(0.f, 0.f, 0.f);
  static const Vector one(1.f, 1.f, 1.f);
  std::vector<Contribution> contributions;
  contributions.reserve(2*tracer.rouletteDepth);
  if (job.pathCapture) {
    job.vertices.clear();
    job.vertices.reserve(contributions.capacity());
  }
  bool hitEmitter = false;
  auto gen = job.getGenerators();

  std::size_t depth = 0;
  const std::size_t maxDepth = 20;

  // Loop to trace the ray through the scence and produce the ray path:
  auto ray = cameraRay;
  while (depth < maxDepth) {
    // Russian roulette ray termination:
    float rrFactor = 1.f;
    if (depth >= tracer.rouletteDepth) {
      bool stop;
      const float rnd1 = xoshiro::uniform_0_1(gen.rng);
      std::tie(stop, rrFactor) = rouletteWeight(rnd1, tracer.stopProb);
      if (stop) { break; }
    }

    // Compute hit point and advance the ray to the intersection:
    const auto intersection = tracer.scene.intersect(ray);
    if (!intersection) { break; }

    if (job.pathCapture) {
      job.vertices.push_back(ray.origin);
    }

    if (intersection.material->emissive) {
      contributions.push_back({intersection.material->emission, rrFactor, Contribution::Type::EMIT});
      hitEmitter = true;
      break;
    }

    if (intersection.material->type == Material::Type::diffuse) {
      const auto rnd1 = xoshiro::uniform_0_1(gen.rng);
      const auto rnd2 = xoshiro::uniform_0_1(gen.rng);
      const auto result = diffuse(ray, intersection.normal, intersection, rrFactor, rnd1, rnd2);
      contributions.push_back(result);
    } else if (intersection.material->type == Material::Type::specular) {
      reflect(ray, intersection.normal);
      contributions.push_back({zero, rrFactor, Contribution::Type::SPECULAR});
    } else if (intersection.material->type == Material::Type::refractive) {
      const auto rnd1 = xoshiro::uniform_0_1(gen.rng);
      refract(ray, intersection.normal, tracer.refractiveIndex, rnd1);
      contributions.push_back({zero, 1.15f * rrFactor, Contribution::Type::REFRACT});
    }

    depth += 1;
  }

  job.totalRayCasts += depth;
  job.maxPathLength = std::max(job.maxPathLength, depth);
  job.nonZeroContribution = hitEmitter;

  // Combine all the material contributions along the ray path:
  Vector total = zero;
  if (hitEmitter) {
    while (!contributions.empty()) {
      auto c = contributions.back();
      contributions.pop_back();

      switch (c.type) {
      case light::Contribution::Type::DIFFUSE:
        // Diffuse materials modulate the colour being carried back
        // along the light path (scaled by the importance weight):
        total = total.cwiseProduct(c.clr) * c.weight;
        break;
        // Emitters add their colour to the colour being carried back
        // along the light path (scaled by the importance weight):
      case light::Contribution::Type::EMIT:
        total += c.clr * c.weight;
        break;
        // Specular reflections/refractions have no colour contribution but
        // their importance sampling weights must still be applied:
      case light::Contribution::Type::SPECULAR:
      case light::Contribution::Type::REFRACT:
        total *= c.weight;
        break;
      // Sometimes it is useful to be able to skip certain
      // contributions when debugging.
      case light::Contribution::Type::SKIP:
      default:
        break;
      }
    }
  }
  return total;
}
