#ifndef AUGLAG_L1_GPU_V_H_
#define AUGLAG_L1_GPU_V_H_

#include "cl.hpp"
#include "R2.h"

namespace auglag1gpu {

class V {
public:
	enum Triangle {odd = 0, even = 1};
	
	V(cl::Context &context, int n1, int n2, int ndf, double h);
	V(const V &a);
	~V();
	V& operator=(const V &a);
	
	V copy(cl::CommandQueue &queue);
	int pull(cl::CommandQueue &queue);
	int push(cl::CommandQueue &queue);
	
	cl::Buffer getHandle();
	
	double operator()(int i, int j) const;
	double& operator()(int i, int j);
	R2 grad(int i, int j, Triangle comp) const;
	
	int getN1() const;
	int getN2() const;
	int getNdf() const;
	double getH() const;
	
private:
	struct Data {
		cl::Context context;
		int refCounter;
		double *hostPtr;
		cl::Buffer devicePtr;
		int n1;
		int n2;
		int ndf;
		double h;
	};
	Data *data;
};

double dot(const V& x1, const V &x2);
double dot(const V& x);
double norm2(const V& x);


}

#endif