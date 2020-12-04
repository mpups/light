#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <string>
#include <vector>
#include <limits>
#include <memory>
#include <cmath>
#include <tuple>
#include <chrono>

#include <boost/program_options.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "light.hpp"
#include "jobs.hpp"
#include "xoshiro.hpp"

XoshiroState rngState = {1654, 4};
light::Halton hal;
light::Halton hal2;

std::pair<bool, float> rouletteWeight(const float stopProb) {
	if (rnd2(rngState) <= stopProb) { return std::make_pair(true, 1.0); }
	return std::make_pair(false, 1.0 / (1.0 - stopProb));
}

light::Vector trace(light::Ray& ray, light::RayTracerContext tracer);

// Diffuse BRDF - choose an outgoing direction with hemisphere sampling.
light::Contribution diffuse(light::Ray& ray, light::Vector normal,
										 const light::Intersection& intersection, float rrFactor) {
	using namespace light;
	Vector rotX, rotY;
	std::tie(rotX, rotY, std::ignore) = orthonormalSystem(normal);

#ifdef USE_EIGEN
	Eigen::Matrix3f R;
	R << rotX, rotY, N;
	ray.direction = R * hemisphere(rnd2(rngState), rnd2(rngState));	// Rotation applied to normalised vector is still unit.
#else
	const auto sampledDir = hemisphere(rnd2(rngState), rnd2(rngState));
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
void refract(light::Ray& ray, light::Vector normal, light::RayTracerContext tracer) {
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
	if (cost2 > 0 && rnd2(rngState) > Rprob) { // refraction direction
		ray.direction = ((ray.direction*n)+(normal*(n*cost1-sqrt(cost2)))).normalized();
	} else { // reflection direction
		ray.direction = (ray.direction+normal*(cost1*2)).normalized();
	}
}

light::Vector trace(light::Ray& ray, light::RayTracerContext tracer) {
	using namespace light;
	static const Vector zero(0, 0, 0);
	static const Vector one(1, 1, 1);
	std::vector<Contribution> contributions;
	contributions.reserve(2*tracer.rouletteDepth);
	bool hitEmitter = false;

	while (true) {
		// Russian roulette ray termination:
		float rrFactor = 1.0;
		if (tracer.depth >= tracer.rouletteDepth) {
			bool stop;
			std::tie(stop, rrFactor) = rouletteWeight(tracer.stopProb);
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
			const auto result = diffuse(ray, normal, intersection, rrFactor);
			contributions.push_back(result);
		} else if (intersection.object->type == Material::specular) {
			reflect(ray, normal);
			contributions.push_back({zero, rrFactor, Contribution::Type::SPECULAR});
		} else if (intersection.object->type == Material::refractive) {
			refract(ray, normal, tracer);
			contributions.push_back({zero, 1.15f * rrFactor, Contribution::Type::REFLECT});
		}

		tracer.next();
	}

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
		("roulette-depth", po::value<float>()->default_value(5), "Number of bounces before rays are randomly stopped.")
		("stop-prob", po::value<float>()->default_value(0.1), "Probability of a ray being stopped.")
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

	light::RayTracerContext tracer;
  tracer.refractiveIndex = args.at("refractive-index").as<float>();
	tracer.rouletteDepth = args.at("roulette-depth").as<float>();
	tracer.stopProb = args.at("stop-prob").as<float>();

	// correlated Halton-sequence dimensions
	hal.number(0, 2);
	hal2.number(0, 2);

  const auto fileName = args.at("outfile").as<std::string>();
  const auto width = args.at("width").as<std::uint32_t>();
  const auto height = args.at("height").as<std::uint32_t>();
	const auto spp = args.at("samples").as<std::uint32_t>();
	const auto antiAliasingScale = args.at("aa-noise-scale").as<float>();
	const auto gui = !args.count("no-gui");

	auto add = [&tracer](Object* o, Vector cl, Vector emission, Material type) {
			o->setMaterial(cl, emission, type);
			tracer.scene.add(o);
	};
	Vector zero(0, 0, 0);
	// Radius, position, color, emission, type (1=diff, 2=spec, 3=refr) for spheres
	add(new Sphere(Vector(-0.75,-1.45,-4.4), 1.05), Vector(4,8,4), zero, Material::specular); // Middle sphere
	add(new Sphere(Vector(2.0,-2.05,-3.7), 0.5), Vector(10,10,1), zero, Material::refractive); // Right sphere
	add(new Sphere(Vector(-1.75,-1.95,-3.1), 0.6), Vector(4,4,12), zero, Material::diffuse); // Left sphere
	// Position, normal, color, emission, type for planes
  const auto X = Vector(1, 0, 0);
  const auto Y = Vector(0, 1, 0);
  const auto Z = Vector(0, 0, 1);
	add(new Plane(Y, 2.5), Vector(6,6,6), zero, Material::diffuse); // Bottom plane
	add(new Plane(Z, 5.5), Vector(6,6,6), zero, Material::diffuse); // Back plane
	add(new Plane(X, 2.75), Vector(10,2,2), zero, Material::diffuse); // Left plane
	add(new Plane(-X, 2.75), Vector(2,10,2), zero, Material::diffuse); // Right plane
	add(new Plane(-Y, 3.0), Vector(6,6,6), zero, Material::diffuse); // Ceiling plane
	add(new Plane(-Z, 0.5), Vector(6,6,6), zero, Material::diffuse); // Front plane
	Vector light1(10000, 5950, 4370);
	add(new Disc(-Y, Vector(0, 2.9999, -4), 0.7), Vector(0,0,0), light1, Material::diffuse); // Ceiling light
	Vector light2(500, 600, 1000);
	add(new Sphere(Vector(-1.12,-2.3,-3.5), 0.2f), Vector(100,200,100), light2, Material::specular); // Small ball light

	light::Image pixels(height);
	for(auto &row: pixels) {
		row.resize(width, Vector(0, 0, 0));
	}

	cv::Mat image(height, width, CV_8UC3);
	image = cv::Vec3b(0, 0, 0);

	auto startTime = std::chrono::steady_clock::now();
  std::uint64_t samples = 0u;
	if (gui) {
		cv::namedWindow(fileName);
	}

	auto jobs = createTracingJobs(width, height, 16, 16, spp);
	for (auto& j : jobs) {
		std::cerr << "Job: " << j << "\n";
	}

	for(std::uint32_t s = 0; s < spp; ++s) {
		double elapsed_secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
		std::cerr << std::setprecision(3) << "\rRendering: " << s*100.f/spp
							<< "% samples: " << samples
							<< " elapsed seconds: " << elapsed_secs
							<< " samples/sec: " << samples / elapsed_secs << "\t";
		#pragma omp parallel for schedule(dynamic) firstprivate(hal, hal2)
		for (std::uint32_t col = 0; col < width; ++col) {
			for(std::uint32_t row = 0; row < height; ++row) {
				Vector cam = camcr(col, row, width, height); // construct image plane coordinates
				Vector aaNoise(rnd(rngState), rnd(rngState), 0.f);
				cam += aaNoise * antiAliasingScale;
				Ray ray(Vector(0, 0, 0), cam);
				auto color = trace(ray, tracer);
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
		if (s == spp - 1 || s % 16 == 0) {
			for (std::uint32_t col = 0; col < width; ++col) {
				for(std::uint32_t row = 0; row < height; ++row) {
					const Vector p = pixels[row][col] * ((spp-1)/(float)s);
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
