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
    ("width,w", po::value<std::uint32_t>()->default_value(256), "Output image width (total pixels).")
    ("height,h", po::value<std::uint32_t>()->default_value(256), "Output image height (total pixels).")
    ("tile-width", po::value<std::uint32_t>()->default_value(16), "Width of tile (pixels).")
    ("tile-height", po::value<std::uint32_t>()->default_value(16), "Height of tile (pixels).")
    ("samples,s", po::value<std::uint32_t>()->default_value(32), "Samples per pixel.")
    ("refractive-index,n", po::value<float>()->default_value(1.5), "Refractive index.")
    ("roulette-depth", po::value<std::size_t>()->default_value(3), "Number of bounces before rays are randomly stopped.")
    ("stop-prob", po::value<float>()->default_value(0.2), "Probability of a ray being stopped.")
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

template <typename SceneType>
void drawDebugRays(const RayDebug& debug,
                   const light::RayTracerContext<SceneType>& tracer,
                   xoshiro::State rngState, cv::Mat& image) {
  using namespace light;
  TraceTileJob debugJob(debug.row, debug.col, debug.row, debug.col, 1,
                        image.cols, image.rows);
  debugJob.pathCapture = true;
  debugJob.rngState = rngState;
  constexpr float fov = light::Pi;
  Vector cam = pixelToRay(debug.col, debug.row, image.cols, image.rows, fov);
  const Ray ray(Vector(0.f, 0.f, 0.f), cam);
  std::size_t maxTries = 100;
  while (--maxTries && debugJob.nonZeroContribution == false) {
    trace(ray, tracer, debugJob);
  }
  std::cerr << "Tries left: " << maxTries << "\n";
  std::vector<Vector> pixels;

  for (auto& v : debugJob.vertices) {
    auto p = vertexToPixel(v, image.cols, image.rows, fov);
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
  using Vec = light::Vector;
  const Vec zero(0.f, 0.f, 0.f);
  const Vec one(1.f, 1.f, 1.f);
  const auto X = Vec(1.f, 0.f, 0.f);
  const auto Y = Vec(0.f, 1.f, 0.f);
  const auto Z = Vec(0.f, 0.f, 1.f);

  // For now we will hard code the scene in the codelet.
  // Eventually the scene will be stored in input tensors:
  light::Sphere spheres[6] = {
    light::Sphere(Vec(-0.75f, -1.45f, -4.4f), 1.05f),
    light::Sphere(Vec(2.0f, -2.05f, -3.7f), 0.5f),
    light::Sphere(Vec(-1.75f, -1.95f, -3.1f), 0.6f),
    light::Sphere(Vec(-1.12f, -2.3f, -3.5f), 0.2f),
    light::Sphere(Vec(-0.28f, -2.34f, -3.f), 0.2f),
    light::Sphere(Vec(.58f, -2.39f, -2.6f), 0.2f)
  };
  light::Plane planes[6] = {
    light::Plane(Y, 2.5f),
    light::Plane(Z, 5.5f),
    light::Plane(X, 2.75f),
    light::Plane(-X, 2.75f),
    light::Plane(-Y, 3.f),
    light::Plane(-Z, .5f)
  };
  light::Disc discs[1] = {
    light::Disc(-Y, Vec(0.f, 2.9999f, -4.f), .7f)
  };

  const Vec lightW(8000, 8000, 8000);
  const Vec lightR(1000, 367, 367);
  const Vec lightG(467, 1000, 434);
  const Vec lightB(500, 600, 1000);
  const float colourGain = 15.f;
  const auto sphereColour = Vec(1.f, .89f, .55f) * colourGain;
  const auto wallColour1 = Vec(.98f, .76f, .66f) * colourGain;
  const auto wallColour2 = Vec(.93f, .43f, .48f) * colourGain;
  const auto wallColour3 = Vec(.27f, .31f, .38f) * colourGain;
  constexpr auto specular = light::Material::Type::specular;
  constexpr auto refractive = light::Material::Type::refractive;
  constexpr auto diffuse = light::Material::Type::diffuse;
  light::Scene<13> scene({
      light::Object(&spheres[0], Vec(4.f, 8.f, 4.f), zero, specular),
      light::Object(&spheres[1], Vec(10.f, 10.f, 1.f), zero, refractive), // Glass sphere
      light::Object(&spheres[2], sphereColour, zero, diffuse), // Diffuse sphere
      light::Object(&spheres[3], zero, lightB, specular), // Small light red
      light::Object(&spheres[4], zero, lightG, specular), // Small light green
      light::Object(&spheres[5], zero, lightR, specular), // Small light blue

      light::Object(&planes[0], wallColour1, zero, diffuse), // Bottom plane
      light::Object(&planes[1], wallColour1, zero, diffuse), // Back plane
      light::Object(&planes[2], wallColour2, zero, diffuse), // Left plane
      light::Object(&planes[3], wallColour3, zero, diffuse), // Right plane
      light::Object(&planes[4], wallColour1, zero, diffuse), // Ceiling plane
      light::Object(&planes[5], wallColour1, zero, diffuse), // Front plane

      light::Object(&discs[0], Vec(0.f, 0.f, 0.f), lightW, diffuse), // Ceiling light
  });

  RayTracerContext<decltype(scene)> tracer(scene);

  tracer.refractiveIndex = args.at("refractive-index").as<float>();
  tracer.rouletteDepth = args.at("roulette-depth").as<std::size_t>();
  tracer.stopProb = args.at("stop-prob").as<float>();
  epsilon = args.at("epsilon").as<float>();
  intersectionEpsilon = args.at("intersection-epsilon").as<float>();
  const auto fileName = args.at("outfile").as<std::string>();
  const auto width = args.at("width").as<std::uint32_t>();
  const auto height = args.at("height").as<std::uint32_t>();
  const auto tileWidth = args.at("tile-width").as<std::uint32_t>();
  const auto tileHeight = args.at("tile-height").as<std::uint32_t>();
  const auto spp = args.at("samples").as<std::uint32_t>();
  const auto antiAliasingScale = args.at("aa-noise-scale").as<float>();
  const auto gui = !args.count("no-gui");

  Image pixels(height);
  for(auto &row: pixels) {
    row.resize(width, Vector(0.f, 0.f, 0.f));
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
      job.visitPixels([&] (std::size_t row, std::size_t col, Vector& p) {
        constexpr float fov = light::Pi/2.f;
        Vector cam = pixelToRay(col, row, width, height, fov); // construct image plane coordinates
        Vector aaNoise(xoshiro::uniform_neg1_1(job.rngState), xoshiro::uniform_neg1_1(job.rngState), 0.f);
        cam += aaNoise * antiAliasingScale;
        const Ray ray(Vector(0.f, 0.f, 0.f), cam);
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
      exr::writeTiled(fileName + ".exr", exr::DataType::FP16,
                      width, height, tileWidth, tileHeight, pixelsFromJobs(jobs));
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
  std::ofstream histFile(fileName + ".hist.txt");
  for (auto& j : jobs) {
    totalRays += j.totalRayCasts;
    maxPathLength = std::max(maxPathLength, j.maxPathLength);
    histFile << j.maxPathLength << "\n";
  }

  auto raysPerSec = totalRays/seconds.count();
  std::cout << "\nTotal Rays: " << totalRays << " Rays/sec: " << raysPerSec << " Max path length: " << maxPathLength << "\n";

  return EXIT_SUCCESS;
}
