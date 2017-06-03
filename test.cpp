/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <stdexcept>
#include <math.h>
#include <limits>
#include <iostream>
#include <algorithm>
#include <fstream>

#include "Image.h"
#include "AuglagL1gpu.h"

using namespace std;

template <typename T>
class RandI {
public:
	static T i(T low, T high) {
		return rand() % (high-low+1) + low;
	}
};

typedef RandI<int> RandInt;

template <typename T>
class RandF {
public:
	static T r(T o, T r) {
		return i(o-r, o+r);
	}
	static T i(T low, T high) {
		return (high-low)*((T)rand()/(T)RAND_MAX) + low;
	}
};

typedef RandF<double> RandDouble;

void imageTest(std::string name, std::string _org, std::string _noise, double r0, double delta, int maxIter) {
	
	Image<double> org(_org);
	Image<double> noise(_noise);
	
	Image<double> noisy = org;
	for(size_t j = 0; j < noisy.getHeight(); j++)
		for(size_t i = 0; i < noisy.getWidth(); i++)
			noisy(i,j) = org(i,j) + noise(i,j) - 0.5;
	
	Image<double> result(noisy.getWidth(), noisy.getHeight());
	Image<double> diff(noisy.getWidth(), noisy.getHeight());

	double h = 0.005;//1.0/max(noise.getWidth()-1, noise.getHeight()-1);
	
	//  5% noise => r0 = 0.001
	// 10% noise => r0 = 0.002
	// 20% noise => r0 = 0.004 - 0.005
	// 40% noise => r0 = 0.020

	double eps = 1.0 * r0 * h;
	double r1 = 10.0 * r0 * h;
	double r2 = 5.0 * r0;
	double r3 = 5.0 * r0 * h*h;

	std::cout << ">>> File: " << _org << " (" << org.getHeight() << " x " << org.getWidth() << ")" << std::endl;
	std::cout << ">>> r0 = " << r0 << std::endl;
	
	//
	// Basic OpenCL stuff
	//
	
	cl_int err;

	std::vector<cl::Platform> platforms;
	err = cl::Platform::get(&platforms);

	if(err != CL_SUCCESS) {
		std::cerr << "<error> unitTests: Cannot get OpenCL platform. " << auglag1gpu::CLErrorMessage(err) << std::endl;
		return;
	}

	if(platforms.size() < 1) {
		std::cerr << "<error> unitTests: Cannot get OpenCL platform." << std::endl;
		return;
	}

	std::vector<cl::Device> devices;
	err = platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

	if(err != CL_SUCCESS) {
		std::cerr << "<error> unitTests: Cannot get OpenCL devices. " << auglag1gpu::CLErrorMessage(err) << std::endl;
		return;
	}

	if(devices.size() < 1) {
		std::cerr << "<error> unitTests: Cannot get OpenCL devices." << std::endl;
		return;
	}

	cl::Device device = devices[RandInt::i(0,devices.size()-1)];
	
	std::vector<cl::Device> oneDevice;
	oneDevice.push_back(device);

	cl::Context context = cl::Context(oneDevice, 0, 0, 0, &err);

	if(err != CL_SUCCESS) {
		std::cerr << "<error> unitTests: Cannot create OpenCL context. " << auglag1gpu::CLErrorMessage(err) << std::endl;
		return;
	}
	
	std::vector<cl::Context> contexts;
	contexts.push_back(context);

	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE, 0);

	auglag1gpu::AuglagL1gpu solver(context, device, &err);
	
	if(err) {
		std::cerr << "<error> Error while initializing solver." << std::endl;
		return;
	}
	
	solver.solve(context, device, queue, noisy.getRaw(), result.getRaw(), noisy.getWidth(), noisy.getHeight(), noisy.getWidth(), r1, r2, r3, eps, h, delta, maxIter);

	std::cout << ">>> Original norm = " << diffNorm(noisy, org) << std::endl;
	std::cout << ">>> New norm      = " << diffNorm(result, org) << std::endl;

	double maxDiff = 0.0;
	for(size_t j = 0; j < noisy.getHeight(); j++)
		for(size_t i = 0; i < noisy.getWidth(); i++)
			maxDiff = max(maxDiff, fabs(result(i,j) - noisy(i,j)));
	
	for(size_t j = 0; j < noisy.getHeight(); j++)
		for(size_t i = 0; i < noisy.getWidth(); i++)
			diff(i,j) =  fabs(result(i,j) - noisy(i,j)) / maxDiff;

	diff.save(name + "_diff.png");
	org.save(name + "_original.png");
	noisy.save(name + "_noise.png", 0.5);
	result.save(name + "_output.png", 0.5);

	std::size_t sCount = 10;

	for(std::size_t i = 0; i < sCount; i++) {
		std::ofstream ofs;
		std::string fileName = name + "_section_org" + auglag1gpu::toString(i) + ".txt";
		ofs.open(fileName.c_str(), std::ofstream::out);
		crossSection(org, ofs, 1, ((1.0/(sCount+1))*(i+1))*result.getHeight(), result.getWidth()-1, ((1.0/(sCount+1))*(i+1))*result.getHeight(), result.getWidth()-2);
		ofs.close();
	}

	for(std::size_t i = 0; i < sCount; i++) {
		std::ofstream ofs;
		std::string fileName = name + "_section_noise" + auglag1gpu::toString(i) + ".txt";
		ofs.open(fileName.c_str(), std::ofstream::out);
		crossSection(noisy, ofs, 1, ((1.0/(sCount+1))*(i+1))*result.getHeight(), result.getWidth()-1, ((1.0/(sCount+1))*(i+1))*result.getHeight(), result.getWidth()-2);
		ofs.close();
	}

	for(std::size_t i = 0; i < sCount; i++) {
		std::ofstream ofs;
		std::string fileName = name + "_section" + auglag1gpu::toString(i) + ".txt";
		ofs.open(fileName.c_str(), std::ofstream::out);
		crossSection(result, ofs, 1, ((1.0/(sCount+1))*(i+1))*result.getHeight(), result.getWidth()-1, ((1.0/(sCount+1))*(i+1))*result.getHeight(), result.getWidth()-2);
		ofs.close();
	}
}

int main(void) {
	srand (time(NULL));

	imageTest(NAME, ORG, NOISE, R0, 1.0E-4, 500);

	return EXIT_SUCCESS;
}
