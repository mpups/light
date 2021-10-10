#pragma once

#include <OpenEXR/ImfConvert.h>

inline
std::vector<half> floatToHalf(const std::vector<float>& data) {
  std::vector<half> result(data.size());
  for (auto i = 0u; i < data.size(); i += 1) {
    result[i] = Imf::floatToHalf(data[i]);
  }
  return result;
}

inline
std::vector<std::vector<half>> floatToHalf(const std::vector<std::vector<float>>& data) {
  std::vector<std::vector<half>> result(data.size());
  #pragma omp parallel for schedule(dynamic)
  for (auto i = 0u; i < data.size(); i += 1) {
    result[i] = floatToHalf(data[i]);
  }
  return result;
}

inline
std::vector<std::uint32_t> floatToUint(const std::vector<float>& data) {
  std::vector<std::uint32_t> result(data.size());
  for (auto i = 0u; i < data.size(); i += 1) {
    result[i] = Imf::floatToUint(data[i]);
  }
  return result;
}

inline
std::vector<std::vector<std::uint32_t>> floatToUint(const std::vector<std::vector<float>>& data) {
  std::vector<std::vector<std::uint32_t>> result(data.size());
  #pragma omp parallel for schedule(dynamic)
  for (auto i = 0u; i < data.size(); i += 1) {
    result[i] = floatToUint(data[i]);
  }
  return result;
}
