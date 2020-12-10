#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <string>

#include <boost/program_options.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "trace.hpp"

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
		("seed", po::value<std::uint64_t>()->default_value(1), "Seed for random number generation.")
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

	const auto seed = args.at("seed").as<std::uint64_t>();
	auto jobs = createTracingJobs(width, height, 16, 16, spp, seed);
	std::cerr << "Job count: " << jobs.size() << "\n";

	for(std::uint32_t s = 0; s < spp; ++s) {
		double elapsed_secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
		std::cerr << std::setprecision(3) << "\rRendering: " << s*100.f/spp
							<< "% samples: " << samples
							<< " elapsed seconds: " << elapsed_secs
							<< " samples/sec: " << samples / elapsed_secs << "\t";

		#pragma omp parallel for schedule(dynamic)
		for (std::size_t j = 0; j < jobs.size(); ++j) {
			auto& job = jobs[j];
			job.visitPixels([&] (std::size_t row, std::size_t col, light::Vector& p) {
				Vector cam = camcr(col, row, width, height); // construct image plane coordinates
				Vector aaNoise(xoshiro::rnd(job.rngState), xoshiro::rnd(job.rngState), 0.f);
				cam += aaNoise * antiAliasingScale;
				Ray ray(Vector(0, 0, 0), cam);
				auto color = trace(ray, tracer, job.getGenerators());
				p += color / spp; // write the contributions
			});

			// We need to service the window's event loop regularly:
			#pragma omp critical
			if (gui && j % 20 == 0) {
				cv::waitKey(1);
			}
		}

		samples += width * height;

		// Save/display image at regular intervals and when done:
		if (s == spp - 1 || s % 16 == 0) {
			for (auto& j : jobs) {
				j.visitPixels([&] (std::size_t row, std::size_t col, light::Vector& p) {
					const Vector v = p * ((spp-1)/(float)s);
					const auto value = cv::Vec3b(
						std::min(v(2), 255.f),
						std::min(v(1), 255.f),
						std::min(v(0), 255.f)
					);
					image.at<cv::Vec3b>(row, col) = value;
				});
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
