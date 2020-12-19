#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <string>

#include <boost/program_options.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "trace.hpp"
#include "exr/exr.hpp"

boost::program_options::variables_map
getArgs(int argc, char** argv) {
  namespace po = boost::program_options;
  po::options_description desc(std::string(argv[0]) + " usage:");
  desc.add_options()
    ("help", "Output help and exit.")
    ("outfile,o", po::value<std::string>()->required(), "Set output file name.")
    ("width,w", po::value<std::uint32_t>()->default_value(256), "Output image width.")
    ("height,h", po::value<std::uint32_t>()->default_value(256), "Output image height.")
    ("tile-width", po::value<std::uint32_t>()->default_value(16), "Width of tile per job.")
    ("tile-height", po::value<std::uint32_t>()->default_value(16), "Height of tile per job.")
    ("samples,s", po::value<std::uint32_t>()->default_value(32), "Samples per pixel.")
    ("refractive-index,n", po::value<float>()->default_value(1.5), "Refractive index.")
		("roulette-depth", po::value<float>()->default_value(5), "Number of bounces before rays are randomly stopped.")
		("stop-prob", po::value<float>()->default_value(0.1), "Probability of a ray being stopped.")
		("aa-noise-scale,a", po::value<float>()->default_value(1.0/700), "Scale for pixel space anti-aliasing noise.")
		("seed", po::value<std::uint64_t>()->default_value(1), "Seed for random number generation.")
		("epsilon", po::value<float>()->default_value(light::epsilon), "Epsilon used in ray tracing, defaults to the machine epsilon.")
		("intersection-epsilon", po::value<float>()->default_value(light::intersectionEpsilon), "Tolerance to avoid re-intersections of out-going rays.")
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

std::vector<std::vector<float>> pixelsFromJobs(std::vector<TraceTileJob>& jobs) {
  std::vector<std::vector<float>> tiles;
  tiles.reserve(jobs.size());
	for (std::size_t j = 0; j < jobs.size(); ++j) {
		const auto tiledPixelCount = 3 * jobs[j].rows() * jobs[j].cols();
		tiles.emplace_back(tiledPixelCount);
	}

	#pragma omp parallel for schedule(dynamic)
	for (std::size_t j = 0; j < jobs.size(); ++j) {
		auto& tile = tiles[j];
		std::size_t c = 0;
		jobs[j].visitPixels([&] (std::size_t, std::size_t, light::Vector& p) {
			tile[c] = p.x;
			tile[c + 1] = p.y;
			tile[c + 2] = p.z;
			c += 3;
		});
  }
  return tiles;
}

void cvImageFromJobs(std::vector<TraceTileJob>& jobs, cv::Mat& image, float scale) {
	#pragma omp parallel for schedule(dynamic)
	for (std::size_t j = 0; j < jobs.size(); ++j) {
		auto& job = jobs[j];
		job.visitPixels([&] (std::size_t row, std::size_t col, light::Vector& p) {
			const light::Vector v = p * scale;
			const auto value = cv::Vec3b(
				std::min(v(2), 255.f),
				std::min(v(1), 255.f),
				std::min(v(0), 255.f)
			);
			image.at<cv::Vec3b>(row, col) = value;
		});
	}
}
struct RayDebug {
	std::size_t row = 0;
	std::size_t col = 0;
	bool enabled = false;
  RayDebug() : row(0), col(0), enabled(false) {}
};

void onMouseClick(int event, int x, int y, int, void* data) {
	RayDebug& debug = *reinterpret_cast<RayDebug*>(data);
	if  (event == cv::EVENT_LBUTTONDOWN) {
		debug.row = y;
		debug.col = x;
		debug.enabled = true;
	}
	if  (event == cv::EVENT_RBUTTONDOWN) {
		debug.enabled = false;
	}
}

void drawDebugRays(const RayDebug& debug, const light::RayTracerContext& tracer,
									 xoshiro::State rngState, cv::Mat& image) {
	using namespace light;
	TraceTileJob debugJob(debug.row, debug.col, debug.row, debug.col, 1);
	debugJob.pathCapture = true;
	debugJob.rngState = rngState;
	Vector cam = pixelToRay(debug.col, debug.row, image.cols, image.rows);
	const Ray ray(Vector(0, 0, 0), cam);
	std::size_t maxTries = 100;
	while (--maxTries && debugJob.nonZeroContribution == false) {
		trace(ray, tracer, debugJob);
	}
	std::cerr << "Tries left: " << maxTries << "\n";
	std::vector<Vector> pixels;

	for (auto& v : debugJob.vertices) {
		auto p = vertexToPixel(v, image.cols, image.rows);
		pixels.push_back(p);
	}

	static const std::vector<cv::Vec3b> colours = {
		cv::Vec3b(25, 255, 255),
		cv::Vec3b(255, 255, 25),
		cv::Vec3b(25, 25, 255),
		cv::Vec3b(255, 25, 255)
	};

	for (std::size_t p = 1; p < pixels.size(); ++p) {
		auto a = cv::Point2f(pixels[p-1].x, pixels[p-1].y);
		auto b = cv::Point2f(pixels[p].x, pixels[p].y);
		cv::line(image, a, b, colours[p % 4], 2, cv::LINE_AA);
	}
}

int main(int argc, char** argv) {
  const auto args = getArgs(argc, argv);

  using namespace light;

	light::Scene scene;
	light::RayTracerContext tracer(scene);
  tracer.refractiveIndex = args.at("refractive-index").as<float>();
	tracer.rouletteDepth = args.at("roulette-depth").as<float>();
	tracer.stopProb = args.at("stop-prob").as<float>();
  light::epsilon = args.at("epsilon").as<float>();
	light::intersectionEpsilon = args.at("intersection-epsilon").as<float>();

  const auto fileName = args.at("outfile").as<std::string>();
  const auto width = args.at("width").as<std::uint32_t>();
  const auto height = args.at("height").as<std::uint32_t>();
  const auto tileWidth = args.at("tile-width").as<std::uint32_t>();
  const auto tileHeight = args.at("tile-height").as<std::uint32_t>();
	const auto spp = args.at("samples").as<std::uint32_t>();
	const auto antiAliasingScale = args.at("aa-noise-scale").as<float>();
	const auto gui = !args.count("no-gui");

	auto add = [&](Object* o, Vector cl, Vector emission, Material type) {
			o->setMaterial(cl, emission, type);
			scene.add(o);
	};

	Vector zero(0, 0, 0);
	// Radius, position, color, emission, type (1=diff, 2=spec, 3=refr) for spheres
	add(new Sphere(Vector(-0.75,-1.45,-4.4), 1.05), Vector(4,8,4), zero, Material::specular); // Mirror sphere
	add(new Sphere(Vector(2.0,-2.05,-3.7), 0.5), Vector(10,10,1), zero, Material::refractive); // Glass sphere
	add(new Sphere(Vector(-1.75,-1.95,-3.1), 0.6), Vector(4,4,12), zero, Material::diffuse); // Diffuse sphere
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

	Image pixels(height);
	for(auto &row: pixels) {
		row.resize(width, Vector(0, 0, 0));
	}

	cv::Mat image(height, width, CV_8UC3);

	auto startTime = std::chrono::steady_clock::now();
  std::uint64_t samples = 0u;
	RayDebug debug;
	if (gui) {
		cv::namedWindow(fileName);
		cv::setMouseCallback(fileName, onMouseClick, &debug);
	}

	const auto seed = args.at("seed").as<std::uint64_t>();
	auto jobs = createTracingJobs(width, height, tileWidth, tileHeight, spp, seed);
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
				Vector cam = pixelToRay(col, row, width, height); // construct image plane coordinates
				Vector aaNoise(xoshiro::uniform_neg1_1(job.rngState), xoshiro::uniform_neg1_1(job.rngState), 0.f);
				cam += aaNoise * antiAliasingScale;
				const Ray ray(Vector(0, 0, 0), cam);
				auto color = trace(ray, tracer, job);
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
		if (s == spp - 1 || s == 1 || s % 64 == 0) {
			cvImageFromJobs(jobs, image, (spp-1)/(float)s);
			writeTiledExr(fileName + ".exr", width, height, tileWidth, tileHeight, pixelsFromJobs(jobs));
			cv::imwrite(fileName, image);
			if (gui) {
				if (debug.enabled) {
					drawDebugRays(debug, tracer, jobs.back().rngState, image);
				}
				cv::imshow(fileName, image);
			}
		}
	}

	std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - startTime;
	std::cout << "\nRender time: " << seconds.count() << " seconds\n";

	std::size_t totalRays = 0;
	std::size_t maxPathLength = 0;
	for (auto& j : jobs) {
		totalRays += j.totalRayCasts;
		maxPathLength = std::max(maxPathLength, j.maxPathLength);
	}

	auto raysPerSec = totalRays/seconds.count();
	std::cout << "\nTotal Rays: " << totalRays << " Rays/sec: " << raysPerSec << " Max path length: " << maxPathLength << "\n";

  return EXIT_SUCCESS;
}
