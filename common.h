#ifndef AUGLAGL1GPU_COMMON_H_
#define AUGLAGL1GPU_COMMON_H_

#include <string>
#include <sstream>
#include <exception>
#include <sys/time.h>

#include "cl.hpp"

namespace auglag1gpu {

// template <typename T>
// class RandI {
// public:
// 	static T i(T low, T high) {
// 		return rand() % (high-low+1) + low;
// 	}
// };
// 
// typedef RandI<int> RandInt;
// 
// template <typename T>
// class RandF {
// public:
// 	static T r(T o, T r) {
// 		return i(o-r, o+r);
// 	}
// 	static T i(T low, T high) {
// 		return (high-low)*((T)rand()/(T)RAND_MAX) + low;
// 	}
// };
	
#ifndef DIVCEIL
int DIVCEIL(int a, int b);
#endif
	
template <typename T>
class ToStringHelper {
public:
	static std::string to(const T& value) {
		std::stringstream ss;
		ss << value;
		return ss.str();
	}
};

template <>
class ToStringHelper<cl::Buffer> {
public:
	static std::string to(const cl::Buffer& value) {
		return "<device address>";
	}
};

template <>
class ToStringHelper<bool> {
public:
	static std::string to(const bool& value) {
		return value ? "true" : "false";
	}
};

template <typename T>
std::string toString(const T& value) {
	return ToStringHelper<T>::to(value);
}

std::string CLErrorMessage(cl_int code);

class Timer {
public:
	Timer() {
		running = false;
		ready = false;
	}
	void begin() {
		gettimeofday(&begin_time, NULL);
		running = true;
		ready = false;
	}
	void end() {
		if(!running)
			throw std::exception();
		gettimeofday(&end_time, NULL);
		running = false;
		ready = true;
	}
	double getTime() const {
		return getEndTime() - getBeginTime();
	}
	double getBeginTime() const {
		if(!running && !ready)
			throw std::exception();
		return begin_time.tv_sec + begin_time.tv_usec*1.0E-6;
	}
	double getEndTime() const {
		if(!ready)
			throw std::exception();
		return end_time.tv_sec + end_time.tv_usec*1.0E-6;
	}
private:
	bool running ;
	bool ready;
	struct timeval begin_time;
	struct timeval end_time;
};

}

#endif