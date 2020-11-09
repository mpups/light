#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <limits>

#include <boost/program_options.hpp>
#include <Eigen/Dense>

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

namespace light {

using Vector = Eigen::Vector3f;

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
	Object(const Vector& c, float e, Material m) : colour(c), emission(e), type(m) {}
  virtual ~Object() {}

	virtual Vector normal(const Vector&) const = 0;
  virtual float intersect(const Ray&) const = 0;

  static float eps;
};

float Object::eps = std::numeric_limits<float>::epsilon();

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
	Vector centre;
	float radius;

	Sphere(Vector c, float r) : centre(c), radius(r) {}
  virtual ~Sphere() {}

	float intersect(const Ray& ray) const override {
		auto b = (2.f * (ray.origin - centre)).dot(ray.direction);
    auto r2 = radius * radius;
		auto c = (ray.origin - centre).dot((ray.origin - centre)) - r2;
		auto disc = b*b - 4*c;
		if (disc<0) {
      return 0.f;
    } else {
      disc = sqrt(disc);
    }
		auto sol1 = -b + disc;
		auto sol2 = -b - disc;
		return (sol2>eps) ? sol2/2 : ((sol1>eps) ? sol1/2 : 0);
	}

	Vector normal(const Vector& point) const {
		return (point - centre).normalized();
	}
};

}

int main(int argc, char** argv) {
  const auto args = getArgs(argc, argv);

  std::cerr << "light::Object epsilon: " << light::Object::eps << "\n";

  Eigen::VectorXf r1(3,1);
  Eigen::VectorXf r2(3,1);
  r1 << 0.1, 0, 0;
  auto ray = light::Ray(r1, r1);
  std::cout << "result: " << ray.direction.transpose() << "\n";
  return EXIT_SUCCESS;
}
