#include <Eigen/Dense>

namespace light {

#ifdef USE_EIGEN
using Vector = Eigen::Vector3f;
#else
struct Vector {
	float x, y, z;
	Vector() {}
	Vector(float x0, float y0, float z0) : x(x0), y(y0), z(z0) {}
	Vector operator + (const Vector &b) const {
		return Vector(x + b.x, y + b.y, z + b.z);
	}
	Vector& operator += (const Vector &b) {
		x += b.x;
		y += b.y;
		z += b.z;
		return *this;
	}
	Vector operator - (const Vector &b) const {
		return Vector(x-b.x,y-b.y,z-b.z);
	}
	Vector operator - () const { return Vector(-x, -y, -z); }
	Vector operator * (float b) const { return Vector(x*b, y*b, z*b); }
	Vector operator / (float b) const { return Vector(x/b, y/b, z/b); }
	Vector cwiseProduct(const Vector &b) const {
		return Vector(x*b.x, y*b.y, z*b.z);
	}
  Vector normalized() { return *this * (1.f/sqrt(x*x + y*y + z*z)); }
	float squaredNorm() { return x*x + y*y + z*z; }
	float dot(const Vector &b) const { return x*b.x + y*b.y + z*b.z; }
	Vector cross(const Vector &b) const {
		return Vector(y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x);
	}
	const float& operator () (size_t i) const { return i==0 ? x : (i==1 ? y : z); }
	Vector abs() const {
		return Vector(std::abs(x), std::abs(y), std::abs(z));
	}
	const Vector& array() const { return *this; } // For Eigen compatibility only
};
#endif

} // end namespace light
