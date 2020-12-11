#pragma once

#include "light.hpp"
#include "xoshiro.hpp"

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
	light::Image pixels;
	xoshiro::State rngState;
	std::size_t totalRayCasts;
	std::size_t maxPathLength;

	TraceTileJob(std::size_t sr, std::size_t sc,
							 std::size_t er, std::size_t ec,
               std::size_t samples)
							 : 	startRow(sr), endRow(er),
							 	  startCol(sc), endCol(ec), spp(samples),
									pixels(endRow - startRow),
                  rngState({0, 0}),
									totalRayCasts(0),
									maxPathLength(0)
	{
		for(auto &row: pixels) {
			row.resize(endCol - startCol, light::Vector(0, 0, 0));
		}
	}

	using Visitor = std::function<void(std::size_t r, std::size_t c, light::Vector& p)>;

	void visitPixels(Visitor&& visit) {
		for (std::size_t r = startRow; r < endRow; ++r) {
			for (std::size_t c = startCol; c < endCol; ++c) {
					visit(r, c, pixels[r - startRow][c - startCol]);
			}
		}
	}

	Generators getGenerators() { return Generators(rngState); }

	std::size_t rows() const { return endRow - startRow; }
	std::size_t cols() const { return endCol - startCol; }
};

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
			jobs.emplace_back(r, c, endRow, endCol, samples);
			jobs.back().rngState = state;
			xoshiro::jump(state);
		}
	}
	return jobs;
}

void accumulateTraceJobResults(std::vector<TraceTileJob>& jobs, light::Image& image) {
	for (auto& j : jobs) {
		j.visitPixels([&] (std::size_t r, std::size_t c, light::Vector& p) {
			image[r][c] += p;
		});
	}
}

std::ostream& operator << (std::ostream &os, const TraceTileJob &j) {
    os << "[" << j.startRow << "," << j.endRow << "), [" << j.startCol << "," << j.endCol << ")";
    return os;
}
