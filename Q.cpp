#include <exception>
#include <iostream>

#include "common.h"
#include "Q.h"

using namespace auglag1gpu;

Q::Q(cl::Context &context, int m1, int m2, int mdf, double h) {
	cl_int err;
	cl::Buffer buffer(context, CL_MEM_READ_WRITE, 4*m2*mdf*sizeof(cl_double), 0, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "<error> Q::Q: Cannot allocate device memory. " << CLErrorMessage(err) << std::endl;
		throw std::exception();
	}
	
	data = new Data;
	data->context = context;
	data->refCounter = 1;
	data->hostPtr = 0;
	data->devicePtr = buffer;
	data->m1 = m1;
	data->m2 = m2;
	data->mdf = mdf;
	data->h = h;
}

Q::Q(const Q &a) {
	data = a.data;
	data->refCounter++;
}

Q::~Q() {
	data->refCounter--;
	if(data->refCounter == 0) {
		if(data->hostPtr)
			delete data->hostPtr;
		delete data;
	}
}

Q& Q::operator=(const Q &a) {
	if(this == &a)
		return *this;
	
	data->refCounter--;
	if(data->refCounter == 0) {
		if(data->hostPtr)
			delete data->hostPtr;
		delete data;
	}
	
	data = a.data;
	data->refCounter++;
	
	return *this;
}

Q Q::hostCopy() {
	Q q(data->context, getM1(), getM2(), getMdf(), getH());
	
	::hostCopy(q, *this);
	
	return q;
}
Q Q::devCopy(cl::CommandQueue &queue, cl_int *err) {
	Q q(data->context, getM1(), getM2(), getMdf(), getH());
	
	::devCopy(queue, q, *this, err);
	
	*err = queue.finish();
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> Q::devCopy: Cannot finish command queue. " << CLErrorMessage(*err) << std::endl;
	}
	
	return q;
}

Q Q::copy(cl::CommandQueue &queue, cl_int *err) {
	Q q(data->context, getM1(), getM2(), getMdf(), getH());
	
	::devCopy(queue, q, *this, err);
	::hostCopy(q, *this);
	
	*err = queue.finish();
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> Q::copy: Cannot finish command queue. " << CLErrorMessage(*err) << std::endl;
	}
	
	return q;
}

int Q::pull(cl::CommandQueue &queue) {
	cl_int err;
	
	int m1 = getM1();
	int m2 = getM2();
	int mdf = getMdf();
	
	double *tmp = new double[4*m2*mdf];
	if(data->hostPtr == 0) 
		data->hostPtr = new R2[2*m2*mdf];
	
	err = queue.enqueueReadBuffer(data->devicePtr, true, 0, 4*m2*mdf*sizeof(cl_double), tmp, 0, 0);
	if(err != CL_SUCCESS) {
		std::cerr << "<error> Q::pull: Cannot read from device memory. " << CLErrorMessage(err) << std::endl;
		delete [] tmp;
		return err;
	}
	
	for(int j = 0; j < m2; j++) {
		for(int i = 0; i < m1; i++) {
			(*this)(i,j,odd)  = R2(tmp[     j*mdf+i], tmp[(2*m2+j)*mdf+i]);
			(*this)(i,j,even) = R2(tmp[(m2+j)*mdf+i], tmp[(3*m2+j)*mdf+i]);
		}
	}
	
	delete [] tmp;
	
	return CL_SUCCESS;
}

int Q::push(cl::CommandQueue &queue) {
	cl_int err;
	
	if(data->hostPtr == 0)
		return -9999;
	
	int m1 = getM1();
	int m2 = getM2();
	int mdf = getMdf();
	
	double *tmp = new double[4*m2*mdf];
	
	for(int j = 0; j < m2; j++) {
		for(int i = 0; i < m1; i++) {
			tmp[       j*mdf+i] = (*this)(i,j,odd).x1;
			tmp[(2*m2+j)*mdf+i] = (*this)(i,j,odd).x2;
			tmp[(  m2+j)*mdf+i] = (*this)(i,j,even).x1;
			tmp[(3*m2+j)*mdf+i] = (*this)(i,j,even).x2;
		}
	}
	
	err = queue.enqueueWriteBuffer(data->devicePtr, true, 0, 4*m2*mdf*sizeof(cl_double), tmp, 0, 0);
	if(err != CL_SUCCESS) {
		std::cerr << "<error> Q::push: Cannot write into device memory. " << CLErrorMessage(err) << std::endl;
		delete [] tmp;
		return err;
	}
	
	delete [] tmp;
	
	return CL_SUCCESS;
}

cl::Buffer Q::getHandle() {
	return data->devicePtr;
}

R2 Q::operator()(int i, int j, Triangle comp) const {
	int m1 = getM1();
	int m2 = getM2();;
	
#if FULL_DEBUG
	if(i < 0 || j < 0 || m1 <= i || m2 <= j) {
		std::cerr << "<error> Q::operator(): Invalid index." << std::endl;
		throw std::exception();
	}
#endif

	if(data->hostPtr == 0)
		return R2(NAN, NAN);
		
	if(comp == odd)
		return data->hostPtr[j*m1+i];
	else
		return data->hostPtr[(m2+j)*m1+i];
}

R2& Q::operator()(int i, int j, Triangle comp) {
	int m1 = getM1();
	int m2 = getM2();
	
#if FULL_DEBUG
	if(i < 0 || j < 0 || m1 <= i || m2 <= j) {
		std::cerr << "<error> Q::operator(): Invalid index." << std::endl;
		throw std::exception();
	}
#endif

	if(data->hostPtr == 0)
		data->hostPtr = new R2[2*m2*getMdf()];
	
	if(comp == odd)
		return data->hostPtr[j*m1+i];
	else
		return data->hostPtr[(m2+j)*m1+i];
}

double Q::div(int i, int j) const {

	if(i == 0 || j == 0 || i == getM1() || j == getM2())
		return 0.0;

	double tmp = 0;

	tmp += dot((*this)(i-1, j,   even), R2( 1.0, -1.0));
	tmp += dot((*this)(i,   j,   odd),  R2( 0.0, -1.0));
	tmp += dot((*this)(i,   j,   even), R2(-1.0,  0.0));

	tmp += dot((*this)(i-1, j-1, odd),  R2( 1.0, 0.0));
	tmp += dot((*this)(i-1, j-1, even), R2( 0.0, 1.0));
	tmp += dot((*this)(i,   j-1, odd),  R2(-1.0, 1.0));

	return - tmp / (2.0 * getH());
}

int Q::getM1() const {
	return data->m1;
}

int Q::getM2() const {
	return data->m2;
}

int Q::getMdf() const {
	return data->mdf;
}

double Q::getH() const {
	return data->h;
}

void auglag1gpu::hostCopy(Q &a, Q&b) {
	if(!b.data->hostPtr && a.data->hostPtr) {
		delete [] a.data->hostPtr;
		a.data->hostPtr = 0;
		return;
	}
	
	if(b.data->hostPtr) {
		int bufferSize = 2*a.getM2()*a.getM1();
		if(!a.data->hostPtr)
			a.data->hostPtr = new R2[bufferSize];
		for(int i = 0; i < bufferSize; i++)
			a.data->hostPtr[i] = b.data->hostPtr[i];
	}
}

void auglag1gpu::devCopy(cl::CommandQueue &queue, Q &a, Q&b, cl_int *err) {
	*err = queue.enqueueCopyBuffer(b.getHandle(), a.getHandle(), 0, 0, 4*a.getM2()*a.getMdf()*sizeof(double), 0, 0);
}

void auglag1gpu::copy(cl::CommandQueue &queue, Q &a, Q&b, cl_int *err) {
	hostCopy(a, b);
	devCopy(queue, a, b, err);
}

double auglag1gpu::dot(const Q& x1, const Q &x2) {
#if FULL_DEBUG
	if(x1.getM1() != x2.getM1() || x1.getM2() != x2.getM2()) {
		std::cerr << "<error> dot(Q,Q): Invalid vector size." << std::endl;
		throw std::exception();
	}
#endif
	
	int m1 = x1.getM1();
	int m2 = x1.getM2();
	
	double dot = 0.0;
	double c = 0.0;
	
	for(int j = 0; j < m2; j++) {
		for(int i = 0; i < m1; i++) {
			double dot2 = 0.0;
			dot2 += x1(i,j,Q::odd).x1 * x2(i,j,Q::odd).x1;
			dot2 += x1(i,j,Q::odd).x2 * x2(i,j,Q::odd).x2;
			dot2 += x1(i,j,Q::even).x1 * x2(i,j,Q::even).x1;
			dot2 += x1(i,j,Q::even).x2 * x2(i,j,Q::even).x2;
			double y = dot2 - c;
			double t = dot + y;
			c = (t - dot) - y;
			dot = t;
		}
	}
	
	return dot;
}

double auglag1gpu::dot(const Q& x) {	
	int m1 = x.getM1();
	int m2 = x.getM2();
	
	double dot = 0.0;
		double c = 0.0;
	
	for(int j = 0; j < m2; j++) {
		for(int i = 0; i < m1; i++) {
			double dot2 = 0.0;
			dot2 += x(i,j,Q::odd).x1 * x(i,j,Q::odd).x1;
			dot2 += x(i,j,Q::odd).x2 * x(i,j,Q::odd).x2;
			dot2 += x(i,j,Q::even).x1 * x(i,j,Q::even).x1;
			dot2 += x(i,j,Q::even).x2 * x(i,j,Q::even).x2;
			double y = dot2 - c;
			double t = dot + y;
			c = (t - dot) - y;
			dot = t;
		}
	}
	
	return dot;
}

double auglag1gpu::norm2(const Q& x) {
	return sqrt(dot(x));
}