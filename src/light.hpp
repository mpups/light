#pragma once

#include <cstdlib>
#include <array>
#include <limits>
#include <memory>
#include <cmath>
#include <tuple>
#include <functional>

#include "vector.hpp"
#include "ArrayStack.hpp"

namespace light {

static constexpr float Pi = 3.14159265358979323846264338327950288f;
#ifdef __POPC__
static constexpr float epsilon = std::numeric_limits<float>::epsilon();
static constexpr float intersectionEpsilon = 1e-5;
#else
extern float epsilon;
extern float intersectionEpsilon;
#endif

inline
std::pair<bool, float> rouletteWeight(float rnd1, const float stopProb) {
  if (rnd1 <= stopProb) { return std::make_pair(true, 1.0); }
  return std::make_pair(false, 1.0 / (1.0 - stopProb));
}

struct Ray {
  Vector origin;
  Vector direction;
  Ray() : origin(0.f, 0.f, 0.f), direction(0.f, 0.f, 0.f) {}
  Ray(const Vector& o, const Vector& d) : origin(o), direction(d.normalized()) {}
};

struct Material {
  enum class Type {
    diffuse, specular, refractive
  };

  Material()
  :
    colour(0.f, 0.f, 0.f),
    emission(0.f, 0.f, 0.f),
    type(Material::Type::diffuse),
    emissive(false) {}

  Material(Vector c, Vector e, Material::Type t)
  :
    colour(c), emission(e), type(t),
    emissive(emission.isNonZero())
  {}

  Vector colour;
  Vector emission;
  Type type;
  bool emissive;
};

struct Primitive {
  Material material;

  Primitive() {}
  ~Primitive() {}

  void setMaterial(Vector c, Vector e, Material::Type m) {
    material = Material(c, e, m);
  }

  static constexpr float nan = std::numeric_limits<float>::quiet_NaN();
  virtual Vector normal(const Vector&) const { return Vector(nan, nan, nan); }
  virtual float intersect(const Ray&) const { return nan;}
  };

struct Intersection {
  const Primitive* object;
  const Material* material;
  float t;
  Intersection() : object(nullptr), t(std::numeric_limits<float>::infinity()) {}
  Intersection(const Primitive* const o, float t) : object(o), material(&o->material), t(t) {}
  operator bool() { return object != nullptr; }
};


struct Object {
  Primitive* object;
  Vector colour;
  Vector emission;
  Material::Type type;
};

struct Scene {
  Scene(std::array<Object, 11> o) : objects(o) {
    for (auto& s : objects) {
      s.object->setMaterial(s.colour, s.emission, s.type);
    }
  }
  ~Scene() {}
  Scene(const Scene&) = delete;

  std::array<Object, 11> objects;

  Intersection intersect(const Ray& ray) const {
    Intersection closestIntersection;
    // Dumb linear search:
    for (std::size_t i = 0; i < objects.max_size(); ++i) {
      auto o = objects[i].object;
      auto t = o->intersect(ray);
      if (t > intersectionEpsilon && t < closestIntersection.t) {
        closestIntersection = Intersection(o, t);
      }
    }
    return closestIntersection;
  }
};

struct Plane : public Primitive {
  Vector n;
  float d;
  Plane(const Vector& normal, float offset) : n(normal.normalized()), d(offset) {}
  ~Plane() {}

  virtual Vector normal(const Vector&) const override { return n; }

  virtual float intersect(const Ray& ray) const override {
    auto angle = n.dot(ray.direction);
    if (angle != 0.f) {
      auto t = -((n.dot(ray.origin)) + d) / angle;
      return (t > epsilon) ? t : 0.f;
    }

    return 0.f;
  }
};

struct Disc : public Primitive {
  Vector n;
  Vector c;
  float d;
  float r2;
  Disc(const Vector& normal, const Vector& centre, float radius)
    : n(normal.normalized()), c(centre), d(std::abs(centre.dot(n))), r2(radius*radius) {}
  ~Disc() {}

  virtual Vector normal(const Vector&) const override { return n; }

  virtual float intersect(const Ray& ray) const override {
    auto angle = n.dot(ray.direction);
    if (angle != 0.f) {
      auto t = -((n.dot(ray.origin)) + d) / angle;
      if (t > epsilon) {
        const auto hitPoint = ray.origin + ray.direction*t;
        auto d2 = (hitPoint - c).squaredNorm();
        if (d2 < r2) {
          return t;
        }
      }
    }

    return 0.f;
  }
};

struct Sphere : public Primitive {
  const Vector centre;
  const float radius;
  const float radius2;

  Sphere(Vector c, float r) : centre(c), radius(r), radius2(r*r) {}
  ~Sphere() {}

  virtual float intersect(const Ray& ray) const override {
    Vector f = centre - ray.origin;
    auto tca = f.dot(ray.direction);
    if (tca < 0.f) { return 0.f; }
    Vector l = centre - (ray.origin + (ray.direction * tca));
    auto l2 = l.squaredNorm();
    if (l2 > radius2) { return 0.f; }
    auto thc = sqrtf(radius2 - l2);
    auto t0 = tca - thc;
    auto t1 = tca + thc;
    if (t0 > t1) { std::swap(t0, t1); }
    if (t0 < 0) {
        t0 = t1;
        if (t0 < 0) { return 0.f; }
    }
    return t0;
  }

  virtual Vector normal(const Vector& point) const override {
    return (point - centre).normalized();
  }
};

inline
Vector pixelToRay(float x, float y, std::uint32_t width, std::uint32_t height) {
  float w = width;
  float h = height;
  float fovx = Pi/4;
  float fovy = (h/w) * fovx;
  auto tanfovx = tan(fovx);
  auto tanfovy = tan(fovy);
  return Vector(((2*x-w)/w) * tanfovx,
        -((2*y-h)/h) * tanfovy,
        -1.0);
}

inline
Vector vertexToPixel(Vector v, std::uint32_t width, std::uint32_t height) {
  float w = width;
  float h = height;
  float fovx = Pi/4;
  float fovy = (h/w) * fovx;
  auto tanfovx = tan(fovx);
  auto tanfovy = tan(fovy);
  auto x = -v.x / v.z;
  auto y = v.y / v.z;
  auto px = (w/2) * ((x/tanfovx) + 1.f);
  auto py = (h/2) * ((y/tanfovy) + 1.f);
  return Vector(px, py, v.z);
}

inline
Vector hemisphere(float u1, float u2) {
  const float r = sqrtf(1.f - u1*u1);
  const float phi = 2 * Pi * u2;
  return Vector(cos(phi)*r, sin(phi)*r, u1);
}

inline
std::tuple<Vector, Vector, Vector>
orthonormalSystem(const Vector& v1) {
    Vector v2(0, 0, 0);
    Vector v1abs = v1.array().abs();
    Vector v1sq = v1.cwiseProduct(v1);
    const auto v1x = v1(0);
    const auto v1y = v1(1);
    const auto v1z = v1(2);
    const auto v1x2 = v1sq(0);
    const auto v1y2 = v1sq(1);
    const auto v1z2 = v1sq(2);
    if (v1abs(0) > v1abs(1)) {
      float invLen = 1.f / std::sqrt(v1x2 + v1z2);
      v2 = Vector(-v1z * invLen, 0.f, v1x * invLen);
    } else {
      float invLen = 1.0f / std::sqrt(v1y2 + v1z2);
      v2 = Vector(0.f, v1z * invLen, -v1y * invLen);
    }
    return std::make_tuple(v2, v1.cross(v2), v1);
}

struct RayTracerContext {
  const Scene& scene;
  int depth;
  float refractiveIndex;
  std::size_t rouletteDepth;
  float stopProb;

  RayTracerContext(const Scene& s) : scene(s), depth(0) {}
};

struct Contribution {
  enum class Type {
    DIFFUSE,
    EMIT,
    SPECULAR,
    REFLECT,
    SKIP
  };
  Vector clr;
  float weight;
  Type type;
};

} // end namespace light
