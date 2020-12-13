#pragma once

#include <cstdlib>
#include <vector>
#include <array>
#include <limits>
#include <memory>
#include <cmath>
#include <tuple>

#include "vector.hpp"

namespace light {

extern float epsilon;
extern float intersectionEpsilon;

struct Ray {
	Vector origin;
  Vector direction;
	Ray() : origin(0.f, 0.f, 0.f), direction(0.f, 0.f, 0.f) {}

	Ray(const Vector& o, const Vector& d) : origin(o), direction(d) {
    direction = direction.normalized();
  }
};

enum class Material {
  diffuse, specular, refractive
};

struct Object {
	Vector colour;
	Vector emission;
	Material type;
	bool emissive;

  Object() :
		colour(0.f, 0.f, 0.f), emission(0.f, 0.f, 0.f),
		type(Material::diffuse), emissive(false) {}

	void setMaterial(Vector c, Vector e, Material m) {
    colour = c;
    emission = e;
		if (emission.x == 0.f && emission.y == 0.f && emission.z == 0.f) {
			emissive = false;
		} else {
			emissive = true;
		}
		type = m;
  }

  virtual ~Object() {}

	virtual Vector normal(const Vector&) const = 0;
  virtual float intersect(const Ray&) const = 0;
};

struct Plane : public Object {
	Vector n;
	float d;
	Plane(const Vector& normal, float offset) : n(normal.normalized()), d(offset) {}
  virtual ~Plane() {}

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

struct Disc : public Object {
	Vector n;
	Vector c;
	float d;
	float r2;
	Disc(const Vector& normal, const Vector& centre, float radius)
		: n(normal.normalized()), c(centre), d(std::abs(centre.dot(n))), r2(radius*radius) {}
  virtual ~Disc() {}

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

struct Sphere : public Object {
	const Vector centre;
	const float radius;
	const float radius2;

	Sphere(Vector c, float r) : centre(c), radius(r), radius2(r*r) {}
  virtual ~Sphere() {}

	float intersect(const Ray& ray) const override {
		Vector L = centre - ray.origin;
		auto tca = L.dot(ray.direction);
		if (tca < 0.f) { return 0.f; }
		auto d2 = L.squaredNorm() - (tca * tca);
		if (d2 > radius2) { return 0.f; }
		auto thc = sqrtf(radius2 - d2);
		auto t0 = tca - thc;
		auto t1 = tca + thc;
		if (t0 > t1) { std::swap(t0, t1); }
		if (t0 < 0) {
				t0 = t1;
				if (t0 < 0) { return 0.f; }
		}
		return t0;
	}

	Vector normal(const Vector& point) const {
		return (point - centre).normalized();
	}
};

struct Intersection {
	const Object* object;
	float t;
	Intersection() : object(nullptr), t(std::numeric_limits<float>::infinity()) {}
	Intersection(const Object* const o, float t) : object(o), t(t) {}
  Intersection& operator = (const Intersection& other) {
    object = other.object;
    t = other.t;
    return *this;
  }
	operator bool() { return object != nullptr; }
};

struct Scene {
	std::vector<Object*> objects;

	void add(Object* object) {
		objects.push_back(object);
	}

	Intersection intersect(const Ray& ray) const {
		Intersection closestIntersection;
    // Dumb linear search:
		for (const auto o: objects) {
			auto t = o->intersect(ray);
			if (t > intersectionEpsilon && t < closestIntersection.t) {
				closestIntersection = Intersection(o, t);
			}
		}
		return closestIntersection;
	}
};

inline
Vector camcr(float x, float y, std::uint32_t width, std::uint32_t height) {
	float w = width;
	float h = height;
	float fovx = M_PI/4;
	float fovy = (h/w) * fovx;
	return Vector(((2*x-w)/w) * tan(fovx),
				-((2*y-h)/h) * tan(fovy),
				-1.0);
}

inline
Vector hemisphere(float u1, float u2) {
	const float r = sqrtf(1.f - u1*u1);
	const float phi = 2 * M_PI * u2;
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
	Scene scene;
	int depth;
	float refractiveIndex;
	float rouletteDepth;
	float stopProb;

	RayTracerContext() : depth(0) {}
	RayTracerContext& next() {
		depth += 1;
		return *this;
	}
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

using Image = std::vector<std::vector<light::Vector>>;

} // end namespace light
