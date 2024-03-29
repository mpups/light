#pragma once

#include "light.hpp"
#include "xoshiro.hpp"

using Image = std::vector<std::vector<light::Vector>>;

struct Generators {
  Generators(xoshiro::State& state)
    : rng(state) {}

  xoshiro::State& rng;
};

struct TraceTileJob {
  std::size_t startRow;
  std::size_t endRow;
  std::size_t startCol;
  std::size_t endCol;
  std::size_t spp;
  std::size_t imageWidth;
  std::size_t imageHeight;
  std::vector<light::Vector> pixels;
  xoshiro::State rngState;
  std::size_t totalRayCasts;
  std::size_t maxPathLength;
  bool pathCapture;
  bool nonZeroContribution;
  std::vector<light::Vector> vertices;

  TraceTileJob(std::size_t sr, std::size_t sc,
         std::size_t er, std::size_t ec,
         std::size_t samples, std::size_t iw, std::size_t ih)
               :   startRow(sr), endRow(er),
                  startCol(sc), endCol(ec),
                  spp(samples),
                  imageWidth(iw), imageHeight(ih),
                  pixels((endRow - startRow) * (endCol - startCol), light::Vector(0, 0, 0)),
                  rngState({0, 0}),
                  totalRayCasts(0),
                  maxPathLength(0),
                  pathCapture(false),
                  nonZeroContribution(false) {}

  using Visitor = std::function<void(std::size_t r, std::size_t c, light::Vector& p)>;

  void visitPixels(Visitor&& visit) {
    std::size_t i = 0u;
    for (std::size_t r = startRow; r < endRow; ++r) {
      for (std::size_t c = startCol; c < endCol; ++c) {
          visit(r, c, pixels[i]);
          i += 1;
      }
    }
  }

  Generators getGenerators() { return Generators(rngState); }

  std::size_t rows() const { return endRow - startRow; }
  std::size_t cols() const { return endCol - startCol; }
};

inline
std::vector<TraceTileJob> createTracingJobs(std::size_t imageWidth, std::size_t imageHeight,
                                            std::size_t tileWidth, std::size_t tileHeight,
                                            std::size_t samples, std::uint64_t seed) {
  // Make sure every job has a unique random sequence derived from same seed:
  xoshiro::State state;
  xoshiro::seed(state, seed);

  // Split the image into tiles and make a job for each:
  std::vector<TraceTileJob> jobs;
  for (std::size_t r = 0; r < imageHeight; r += tileHeight) {
    for (std::size_t c = 0; c < imageWidth; c += tileWidth) {
      std::size_t endRow = r + tileHeight;
      std::size_t endCol = c + tileWidth;
      if (endRow > imageHeight) {
        endRow = imageHeight;
      }
      if (endCol > imageWidth) {
        endCol = imageWidth;
      }
      jobs.emplace_back(r, c, endRow, endCol, samples, imageWidth, imageHeight);
      jobs.back().rngState = state;
      xoshiro::jump(state);
    }
  }
  return jobs;
}

inline
void accumulateTraceJobResults(std::vector<TraceTileJob>& jobs, Image& image) {
  for (auto& j : jobs) {
    j.visitPixels([&] (std::size_t r, std::size_t c, light::Vector& p) {
      image[r][c] += p;
    });
  }
}

inline
std::ostream& operator << (std::ostream &os, const TraceTileJob &j) {
    os << "[" << j.startRow << "," << j.endRow << "), [" << j.startCol << "," << j.endCol << ")";
    return os;
}
