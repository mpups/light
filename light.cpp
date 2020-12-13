#include "light.hpp"

namespace light {

float epsilon = std::numeric_limits<float>::epsilon();

// The interseciton epsilon is a temprary hack. Need to use stable
// intersection and ray casting algorithms instead.
float intersectionEpsilon = 1e-5;

} // end namespace light
