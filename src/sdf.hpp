#pragma once

#include "light.hpp"

namespace light {

// Diffuse BRDF - choose an outgoing direction with hemisphere sampling.
inline
light::Contribution diffuse(light::Ray& ray, light::Vector normal,
                     const light::Intersection& intersection, float rrFactor,
                     float rnd1, float rnd2) {
  using namespace light;
  Vector rotX, rotY;
  std::tie(rotX, rotY, std::ignore) = orthonormalSystem(normal);

#ifdef USE_EIGEN
  Eigen::Matrix3f R;
  R << rotX, rotY, N;
  ray.direction = R * hemisphere(rnd1, rnd2);  // Rotation applied to normalised vector is still unit.
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
inline
void reflect(light::Ray& ray, light::Vector normal) {
  auto cost = ray.direction.dot(normal);
  ray.direction = (ray.direction - normal * (cost * 2.f)).normalized();
}

// Glass/refractive BRDF - we use the vector version of Snell's law and Fresnel's law
// to compute the outgoing reflection and refraction directions and probability weights.
// Returns true if the ray was refracted.
inline
bool refract(light::Ray& ray, light::Vector normal,
             float ri, float rnd1) {
  auto R0 = (1.f - ri)/(1.f + ri);
  R0 = R0 * R0;
  if(normal.dot(ray.direction) > 0.f) { // we're inside the medium
    normal = -normal;
  } else {
    ri = 1.f / ri;
  }
  auto cost1 = -normal.dot(ray.direction); // cosine of theta_1
  auto cost2 = 1.f - ri * ri * (1.f - cost1 * cost1); // cosine of theta_2
  auto schlickBase = 1.f - cost1;
  auto schlickBase2 = schlickBase * schlickBase;
  auto Rprob = R0 + (1.f - R0) * (schlickBase2 * schlickBase * schlickBase2); // Schlick-approximation
  if (cost2 > 0.f && rnd1 > Rprob) { // refraction direction
    ray.direction = ((ray.direction * ri) + (normal*(ri*cost1 - sqrtf(cost2)))).normalized();
    return true;
  } else { // reflection direction
    ray.direction = (ray.direction + normal*(cost1*2.f)).normalized();
    return false;
  }
}

} // end namespace light
