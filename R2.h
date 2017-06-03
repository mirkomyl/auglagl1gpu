/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef AUGLAGL1GPU_R2_H
#define AUGLAGL1GPU_R2_H

#include <cmath>
#include "common.h"

namespace auglag1gpu {

// R^2-vector
struct R2 {
	R2() {
		x1 = x2 = 0.0;
	}
	R2(double in) {
		x1 = in;
		x2 = in;
	}
	R2(double in1, double in2) {
		x1 = in1;
		x2 = in2;
	}
	R2& operator=(double in) {
		x1 = in;
		x2 = in;
		return *this;
	}
	R2& operator+=(const R2& a) {
		x1 += a.x1;
		x2 += a.x2;
		return *this;
	}
	R2& operator-=(const R2& a) {
		x1 -= a.x1;
		x2 -= a.x2;
		return *this;
	}
	template <typename Y>
	R2& operator*=(Y coef) {
		x1 *= coef;
		x2 *= coef;
		return *this;
	}
	template <typename Y>
	R2& operator/=(Y coef) {
		x1 /= coef;
		x2 /= coef;
		return *this;
	}
	bool operator==(const R2& a) const {
		return x1 == a.x1 && x2 == a.x2;
	}
	bool operator!=(const R2& a) const {
		return x1 != a.x1 || x2 != a.x2;
	}
	double x1;
	double x2;
};

inline R2 operator*(double coef, const R2& a) {
	return R2(coef*a.x1, coef*a.x2);
}

inline R2 operator*(const R2& a, double coef) {
	return coef*a;
}

inline R2 operator/(const R2& a, double coef) {
	return R2(a.x1/coef, a.x2/coef);
}

inline R2 operator+(const R2& a, const R2& b) {
	return R2(a.x1+b.x1, a.x2+b.x2);
}

inline R2 operator-(const R2& a, const R2& b) {
	return R2(a.x1-b.x1, a.x2-b.x2);
}

inline double dot(const R2& a, const R2& b) {
	return a.x1 * b.x1 + a.x2 * b.x2;
}

inline double dot(const R2& a) {
	return dot(a,a);
}

inline double norm(const R2& a) {
	return sqrt(dot(a));
}

inline bool isNaN(const R2& value) {
	return isNaN(value.x1) || isNaN(value.x2);
}

inline bool isInf(const R2& value) {
	return isInf(value.x1) || isInf(value.x2);
}

template <>
class ToStringHelper<R2> {
public:
	static std::string to(const R2& value) {
		return "(" + toString(value.x1) + ", " + toString(value.x2) + ")";
	}
};

}

#endif
