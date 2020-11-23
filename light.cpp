#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <string>
#include <limits>
#include <memory>
#include <cmath>
#include <tuple>
#include <random>
#include <chrono>

#include <boost/program_options.hpp>
#include <Eigen/Dense>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

namespace light {

using Vector = Eigen::Vector3f;
static float eps = std::numeric_limits<float>::epsilon();

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
	float emission;
	Material type;
  Object() : colour(0.f, 0.f, 0.f), emission(0.f), type(Material::diffuse) {}

	void setMaterial(const Vector& c, float e, Material m) {
    colour = c;
    emission = e;
    type = m;
  }

  virtual ~Object() {}

	virtual Vector normal(const Vector&) const = 0;
  virtual float intersect(const Ray&) const = 0;
};

struct Plane : public Object {
	Vector n;
	float d;
	Plane(const Vector& normal, float offset) : n(normal), d(offset) {}
  virtual ~Plane() {}

	virtual Vector normal(const Vector&) const override { return n; }

	virtual float intersect(const Ray& ray) const override {
		auto angle = n.dot(ray.direction);
		if (angle != 0.f) {
			auto t = -((n.dot(ray.origin)) + d) / angle;
			return (t > eps) ? t : 0.f;
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
			if (t > eps && t < closestIntersection.t) {
				closestIntersection = Intersection(o, t);
			}
		}
		return closestIntersection;
	}
};

Vector camcr(float x, float y, std::uint32_t width, std::uint32_t height) {
	float w = width;
	float h = height;
	float fovx = M_PI/4;
	float fovy = (h/w) * fovx;
	return Vector(((2*x-w)/w) * tan(fovx),
				-((2*y-h)/h) * tan(fovy),
				-1.0);
}

Vector hemisphere(float u1, float u2) {
	const float r = sqrtf(1.f - u1*u1);
	const float phi = 2 * M_PI * u2;
	return Vector(cos(phi)*r, sin(phi)*r, u1);
}

struct Halton {
	float value, inv_base;

	void number(int i,int base) {
		float f = inv_base = 1.0 / base;
		value = 0.0;
		while(i > 0) {
			value += f * (float)(i%base);
			i /= base;
			f *= inv_base;
		}
	}

	void next() {
		float r = 1.0 - value - 0.0000001;
		if(inv_base<r) value += inv_base;
		else {
			float h = inv_base, hh;
			do {hh = h; h *= inv_base;} while(h >=r);
			value += hh + h - 1.0;
		}
	}
	float get() { return value; }
};

Eigen::Matrix3f orthonormalSystem(const Vector& v1) {
    Vector v2;
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
		Eigen::Matrix3f m; m << v2, v1.cross(v2), v1;
    return m;
}

std::random_device rd;
std::mt19937 mersenneTwister(rd());
std::uniform_real_distribution<float> uniform;

auto rnd = [&] () {
	return 2.f*uniform(mersenneTwister) - 1.f;
};

auto rnd2 = [&] () {
	return uniform(mersenneTwister);
};

Vector trace(Ray &ray, const Scene& scene, int depth, float refractiveIndex, Halton& hal, Halton& hal2) {
	Vector clr(0, 0, 0);

	// Russian roulette: starting at depth 5, each recursive step will stop with a probability of 0.1
	float rrFactor = 1.0;
	if (depth >= 5) {
		const float rrStopProbability = 0.1;
		if (rnd2() <= rrStopProbability) {
			return clr;
		}
		rrFactor = 1.0 / (1.0 - rrStopProbability);
	}

	Intersection intersection = scene.intersect(ray);
	if (!intersection) return clr;

	// Travel the ray to the hit point where the closest object lies and compute the surface normal there.
	Vector hp = ray.origin + ray.direction * intersection.t;
	Vector N = intersection.object->normal(hp);
	ray.origin = hp;
	// Add the emission, the L_e(x,w) part of the rendering equation, but scale it with the Russian Roulette
	// probability weight.
	const auto emission = intersection.object->emission;
	clr += Vector(emission, emission, emission) * rrFactor;

	// Diffuse BRDF - choose an outgoing direction with hemisphere sampling.
	if(intersection.object->type == Material::diffuse) {
		const Eigen::Matrix3f R = orthonormalSystem(N);
		ray.direction = R * hemisphere(rnd2(), rnd2());	// Rotation applied to normalised vector is still unit.
		float cost = ray.direction.dot(N);
		auto result = trace(ray, scene, depth + 1, refractiveIndex, hal, hal2);
		clr += (result.cwiseProduct(intersection.object->colour)) * cost * 0.1 * rrFactor;
	}

	// Specular BRDF - this is a singularity in the rendering equation that follows
	// delta distribution, therefore we handle this case explicitly - one incoming
	// direction -> one outgoing direction, that is, the perfect reflection direction.
	if(intersection.object->type == Material::specular) {
		auto cost = ray.direction.dot(N);
		ray.direction = (ray.direction - N * (cost * 2.f)).normalized();
		auto result = trace(ray, scene, depth + 1, refractiveIndex, hal, hal2);
		clr += result * rrFactor;
	}

	// Glass/refractive BRDF - we use the vector version of Snell's law and Fresnel's law
	// to compute the outgoing reflection and refraction directions and probability weights.
	if(intersection.object->type == Material::refractive) {
		auto n = refractiveIndex;
		auto R0 = (1.0-n)/(1.0+n);
		R0 = R0*R0;
		if(N.dot(ray.direction) > 0) { // we're inside the medium
			N = -N;
			n = 1 / n;
		}
		n = 1 / n;
		auto cost1 = (N.dot(ray.direction))*-1; // cosine of theta_1
		auto cost2 = 1.0 - n*n*(1.0-cost1*cost1); // cosine of theta_2
		auto Rprob = R0 + (1.0-R0) * powf(1.0 - cost1, 5.0); // Schlick-approximation
		if (cost2 > 0 && rnd2() > Rprob) { // refraction direction
			ray.direction = ((ray.direction*n)+(N*(n*cost1-sqrt(cost2)))).normalized();
		} else { // reflection direction
			ray.direction = (ray.direction+N*(cost1*2)).normalized();
		}

		auto result = trace(ray, scene, depth + 1, refractiveIndex, hal, hal2);
		clr += result * (1.15 * rrFactor);
	}

	return clr;
}

} // end namespace light

boost::program_options::variables_map
getArgs(int argc, char** argv) {
  namespace po = boost::program_options;
  po::options_description desc(std::string(argv[0]) + " usage:");
  desc.add_options()
    ("help", "Output help and exit.")
    ("outfile,o", po::value<std::string>()->required(), "Set output file name.")
    ("width,w", po::value<std::uint32_t>()->default_value(200), "Output image width.")
    ("height,h", po::value<std::uint32_t>()->default_value(200), "Output image height.")
    ("samples,s", po::value<std::uint32_t>()->default_value(32), "Samples per pixel.")
    ("refractive-index,n", po::value<float>()->default_value(1.5), "Refractive index.")
		("aa-noise-scale,a", po::value<float>()->default_value(1.0/700), "Scale for pixel space anti-aliasing noise.")
		("no-gui", "Disable display of render window.")
  ;

  po::variables_map vars;
  auto parsed = po::parse_command_line(argc, argv, desc);
  po::store(parsed, vars);
  if (vars.count("help")) {
    std::cout << desc << "\n";
    exit(-1);
  }
  po::notify(vars);

  std::ofstream optsFile(vars.at("outfile").as<std::string>() + ".command_line_options");
  for (const auto& opt: parsed.options) {
    if (!opt.unregistered) {
      optsFile << opt.string_key << " : ";
      for (auto c: opt.value) {
        optsFile << c << " ";
      }
    }
    optsFile << "\n";
  }

  return vars;
}

int main(int argc, char** argv) {
  const auto args = getArgs(argc, argv);
  std::cerr << "light epsilon: " << light::eps << "\n";

  using namespace light;
  Scene scene;
	auto add = [&scene](Object* s, Vector cl, float emission, Material type) {
			s->setMaterial(cl, emission, type);
			scene.add(s);
	};

	// // Radius, position, color, emission, type (1=diff, 2=spec, 3=refr) for spheres
	add(new Sphere(Vector(-0.75,-1.45,-4.4), 1.05), Vector(4,8,4), 0, Material::specular); // Middle sphere
	//add(new Sphere(Vector(1.5,1.5,-4.4), 1.f), Vector(10,2,1), 0, Material::specular); // High sphere
	add(new Sphere(Vector(2.0,-2.05,-3.7), 0.5), Vector(10,10,1), 0, Material::refractive); // Right sphere
	add(new Sphere(Vector(-1.75,-1.95,-3.1), 0.6), Vector(4,4,12), 0, Material::diffuse); // Left sphere
	// Position, normal, color, emission, type for planes
  const auto X = Vector(1, 0, 0);
  const auto Y = Vector(0, 1, 0);
  const auto Z = Vector(0, 0, 1);
	add(new Plane(Y, 2.5), Vector(6,6,6), 0, Material::diffuse); // Bottom plane
	add(new Plane(Z, 5.5), Vector(6,6,6), 0, Material::diffuse); // Back plane
	add(new Plane(X, 2.75), Vector(10,2,2), 0, Material::diffuse); // Left plane
	add(new Plane(-X, 2.75), Vector(2,10,2), 0, Material::diffuse); // Right plane
	add(new Plane(-Y, 3.0), Vector(6,6,6), 0, Material::diffuse); // Ceiling plane
	add(new Plane(-Z, 0.5), Vector(6,6,6), 0, Material::diffuse); // Front plane
	add(new Sphere(Vector(0,1.9,-3), 0.5), Vector(0,0,0), 10000, Material::diffuse); // Light

  const auto fileName = args.at("outfile").as<std::string>();
  const auto width = args.at("width").as<std::uint32_t>();
  const auto height = args.at("height").as<std::uint32_t>();
	const auto spp = args.at("samples").as<std::uint32_t>();
  const auto refractiveIndex = args.at("refractive-index").as<float>();
	const auto antiAliasingScale = args.at("aa-noise-scale").as<float>();
	const auto gui = !args.count("no-gui");

	std::vector<std::vector<Vector>> pixels(height);
	for(auto &row: pixels) {
		row.resize(width, Vector(0, 0, 0));
	}

	cv::Mat image(height, width, CV_8UC3);
	image = cv::Vec3b(0, 0, 0);

	// correlated Halton-sequence dimensions
	Halton hal, hal2;
	hal.number(0, 2);
	hal2.number(0, 2);

	auto startTime = std::chrono::steady_clock::now();
  std::uint64_t samples = 0u;
	if (gui) {
		cv::namedWindow(fileName);
	}

	for(std::uint32_t s = 0; s < spp; ++s) {
		double elapsed_secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
		std::cerr << std::setprecision(3) << "\rRendering: " << s*100.f/spp
							<< "% samples: " << samples
							<< " elapsed seconds: " << elapsed_secs
							<< " samples/sec: " << samples / elapsed_secs << "\t";
		#pragma omp parallel for schedule(dynamic) firstprivate(hal,hal2)
		for (std::uint32_t col = 0; col < width; ++col) {
			for(std::uint32_t row = 0; row < height; ++row) {
				Vector cam = camcr(col, row, width, height); // construct image plane coordinates
				Vector aaNoise(rnd(), rnd(), 0.f);
				cam += aaNoise * antiAliasingScale;
				Ray ray(Vector(0, 0, 0), cam);
				auto color = trace(ray, scene, 0, refractiveIndex, hal, hal2);
				pixels[row][col] = pixels[row][col] + color / spp; // write the contributions
			}

			if (gui && col % 20 == 0) {
			  // We need to service the window's event loop regularly:
				#pragma omp critical
				cv::waitKey(1);
			}
		}

		samples += width * height;

		// Save/display image at regular intervals and when done:
		if (s == spp || s % 8 == 0) {
			for (std::uint32_t col = 0; col < width; ++col) {
				for(std::uint32_t row = 0; row < height; ++row) {
					const Vector p = pixels[row][col] * (spp/(float)s);
					const auto value = cv::Vec3b(
						std::min(p(2), 255.f),
						std::min(p(1), 255.f),
						std::min(p(0), 255.f)
					);
					image.at<cv::Vec3b>(row, col) = value;
				}
			}

			cv::imwrite(fileName, image);
			if (gui) {
				cv::imshow(fileName, image);
			}

		}
	}

	std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - startTime;
	std::cout << "\nRender time: " << seconds.count() << " seconds\n";

  return EXIT_SUCCESS;
}
