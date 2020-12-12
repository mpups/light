#include "exr.hpp"

void writeTiledExr(std::string fileName,
									 std::size_t imageWidth, std::size_t imageHeight,
									 std::size_t tileWidth, std::size_t tileHeight,
									 const std::vector<std::vector<float>>& tiledPixels) {
	using namespace Imf;
	Header header(imageWidth, imageHeight);
	header.channels().insert("R", Channel(FLOAT));
	header.channels().insert("G", Channel(FLOAT));
	header.channels().insert("B", Channel(FLOAT));
	header.setTileDescription(TileDescription(tileWidth, tileHeight, ONE_LEVEL));

	std::vector<FrameBuffer> framebuffers;
	framebuffers.resize(tiledPixels.size());

	#pragma omp parallel for schedule(dynamic)
	for (std::size_t t = 0; t < tiledPixels.size(); ++t) {
		auto& tile = tiledPixels[t];
		auto& fb = framebuffers[t];
		auto colStride = 3 * sizeof(float);
		auto rowStride = 3 * tileWidth * sizeof(float);
		fb.insert("R", Slice(FLOAT, (char*)tile.data(), colStride, rowStride, 1, 1, 0.0, true, true));
		fb.insert("G", Slice(FLOAT, (char*)(tile.data() + 1), colStride, rowStride, 1, 1, 0.0, true, true));
		fb.insert("B", Slice(FLOAT, (char*)(tile.data() + 2), colStride, rowStride, 1, 1, 0.0, true, true));
	}

	TiledOutputFile out(fileName.c_str(), header);
	std::size_t tx = 0;
	std::size_t ty = 0;
	std::size_t tilesX = imageWidth / tileWidth;
	std::size_t tiles = framebuffers.size();

	for (std::size_t t = 0; t < tiles; ++t) {
		out.setFrameBuffer(framebuffers[t]);
		out.writeTiles(tx, tx, ty, ty);
		tx += 1;
		if (tx >= tilesX) {
			tx = 0;
			ty += 1;
		}
	}
}