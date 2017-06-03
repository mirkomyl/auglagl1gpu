#include <iostream>
#include "V.h"

using namespace auglag1gpu;

V::V(cl::Context &context, int n1, int n2, int ndf, double h) {
	cl_int err;
	
	cl::Buffer buffer(context, CL_MEM_READ_WRITE, n2*ndf*sizeof(cl_double), 0, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "<error> V::V: Cannot allocate device memory. " << CLErrorMessage(err) << std::endl;
		throw std::exception();
	}
	
	data = new Data;
	data->context = context;
	data->refCounter = 1;
	data->hostPtr = new double[n2*ndf];
	data->devicePtr = buffer;
	data->n1 = n1;
	data->n2 = n2;
	data->ndf = ndf;
	data->h = h;
}

V::V(const V &a) {
	data = a.data;
	data->refCounter++;
}

V::~V() {
	data->refCounter--;
	if(data->refCounter == 0) {
		delete data->hostPtr;
		delete data;
	}
}

V& V::operator=(const V &a) {
	if(&a == this)
		return *this;
	
	data->refCounter--;
	if(data->refCounter == 0) {
		delete data->hostPtr;
		delete data;
	}
	
	data = a.data;
	data->refCounter++;
	
	return *this;
}

V V::copy(cl::CommandQueue &queue) {
	V v(data->context, getN1(), getN2(), getNdf(), getH());
	
	int bufferSize = getN2()*getNdf();
	
	queue.enqueueCopyBuffer(getHandle(), v.getHandle(), 0, 0, bufferSize*sizeof(double), 0, 0);
	
	for(int i = 0; i < bufferSize; i++)
		v.data->hostPtr[i] = data->hostPtr[i];
	
	return v;
}

int V::pull(cl::CommandQueue &queue) {
	cl_int err;
	err = queue.enqueueReadBuffer(data->devicePtr, true, 0, getN2()*getNdf()*sizeof(cl_double), data->hostPtr, 0, 0);
	if(err != CL_SUCCESS) {
		std::cerr << "<error> V::pull: Cannot read from device memory. " << CLErrorMessage(err) << std::endl;
	}
	
	return err;
}

int V::push(cl::CommandQueue &queue) {
	cl_int err;
	err = queue.enqueueWriteBuffer(data->devicePtr, true, 0, getN2()*getNdf()*sizeof(cl_double), data->hostPtr, 0, 0);
	if(err != CL_SUCCESS) {
		std::cerr << "<error> V::push: Cannot write into device memory. " << CLErrorMessage(err) << std::endl;
	}
	
	return err;
}

cl::Buffer V::getHandle() {
	return data->devicePtr;
}

double V::operator()(int i, int j) const {
#if FULL_DEBUG
	if(i < 0 || j < 0 || getN1() <= i || getN2() <= j) {
		std::cerr << "<error> V::operator(): Invalid index." << std::endl;
		throw std::exception();
	}
#endif
	
	return data->hostPtr[j*getNdf()+i];
}

double& V::operator()(int i, int j) {
#if FULL_DEBUG
	if(i < 0 || j < 0 || getN1() <= i || getN2() <= j) {
		std::cerr << "<error> V::operator(): Invalid index." << std::endl;
		throw std::exception();
	}
#endif
	
	return data->hostPtr[j*getNdf()+i];
}

R2 V::grad(int i, int j, Triangle comp) const {
	if(comp == odd) {
		return R2((*this)(i+1,j+1)-(*this)(i,j+1), (*this)(i,j+1)-(*this)(i,j))/getH();
	} else {
		return R2((*this)(i+1,j)-(*this)(i,j), (*this)(i+1,j+1)-(*this)(i+1,j))/getH();
	}
}

int V::getN1() const {
	return data->n1;
}

int V::getN2() const {
	return data->n2;
}

int V::getNdf() const {
	return data->ndf;
}

double V::getH() const {
	return data->h;
}

double auglag1gpu::dot(const V& x1, const V &x2) {
	int n1 = x1.getN1();
	int n2 = x1.getN2();
	
	double res = 0.0;
	
	for(int j = 0; j < n2; j++)
		for(int i = 0; i < n1; i++)
			res += x1(i,j) * x2(i,j);
		
	return res;
}
double auglag1gpu::dot(const V& x) {
	int n1 = x.getN1();
	int n2 = x.getN2();
	
	double res = 0.0;
	
	for(int j = 0; j < n2; j++)
		for(int i = 0; i < n1; i++)
			res += x(i,j) * x(i,j);
		
	return res;
}

double auglag1gpu::norm2(const V& x) {
	return sqrt(dot(x));
}
