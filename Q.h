#ifndef AUGLAG_L1_GPU_Q_H_
#define AUGLAG_L1_GPU_Q_H_

#include "cl.hpp"
#include "R2.h"

namespace auglag1gpu {

class Q {
	friend void hostCopy(Q &a, Q&b);
	friend void devCopy(cl::CommandQueue &queue, Q &a, Q &b, cl_int *err);
	friend void copy(cl::CommandQueue &queue, Q &a, Q &b, cl_int *err);
public:
	enum Triangle {odd = 0, even = 1};
	
	Q(cl::Context &context, int m1, int m2, int mdf, double h);
	Q(const Q &a);
	~Q();
	Q& operator=(const Q &a);
	
	Q hostCopy();
	Q devCopy(cl::CommandQueue &queue, cl_int *err);
	Q copy(cl::CommandQueue &queue, cl_int *err);
	
	int pull(cl::CommandQueue &queue);
	int push(cl::CommandQueue &queue);
	
	cl::Buffer getHandle();
	
	R2 operator()(int i, int j, Triangle comp) const;
	R2& operator()(int i, int j, Triangle comp);
	double div(int i, int j) const;
	
	int getM1() const;
	int getM2() const;
	int getMdf() const;
	double getH() const;
private:
	struct Data {
		cl::Context context;
		int refCounter;
		R2 *hostPtr;
		cl::Buffer devicePtr;
		int m1;
		int m2;
		int mdf;
		double h;
	};
	Data *data;
};

void hostCopy(Q &a, Q &b);
void devCopy(cl::CommandQueue &queue, Q &a, Q &b, cl_int *err);
void copy(cl::CommandQueue &queue, Q &a, Q &b, cl_int *err);

double dot(const Q& x1, const Q &x2);
double dot(const Q& x);
double norm2(const Q& x);

}

#endif