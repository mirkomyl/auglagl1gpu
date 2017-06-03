#include <iostream>
#include <string>

#include <vector>
#include <cmath>

#include "common.h"
#include "AuglagL1gpu.h"

#include "auglagL1gpu.cl.dat"

#if FULL_DEBUG
#ifndef DEBUG
#define DEBUG 1
#else
#if !DEBUG
#undef DEBUG
#define DEBUG 1
#endif
#endif
#endif

#define KARGS_GSUM_SUM			0
#define KARGS_GSUM_LWORK		1
#define KARGS_GSUM_SIZE			2
#define KARGS_GSUM_PART_EXP		3
#define KARGS_GSUM_JUMP_EXP		4

#define KARGS_QDOT_DOT			0
#define KARGS_QDOT_X1			1
#define KARGS_QDOT_X2			2
#define KARGS_QDOT_LWORK		3
#define KARGS_QDOT_M1			4
#define KARGS_QDOT_M2			5
#define KARGS_QDOT_MDF			6

#define KARGS_QDOT_S_DOT		0
#define KARGS_QDOT_S_X			1
#define KARGS_QDOT_S_LWORK		2
#define KARGS_QDOT_S_M1			3
#define KARGS_QDOT_S_M2			4
#define KARGS_QDOT_S_MDF		5

#define KARGS_QSAXPY_Y			0
#define KARGS_QSAXPY_X1			1
#define KARGS_QSAXPY_A			2
#define KARGS_QSAXPY_X2			3
#define KARGS_QSAXPY_M1			4
#define KARGS_QSAXPY_M2			5
#define KARGS_QSAXPY_MDF		6

#define KARGS_QSAXPY_V_Y		0
#define KARGS_QSAXPY_V_X1		1
#define KARGS_QSAXPY_V_A		2
#define KARGS_QSAXPY_V_X2		3
#define KARGS_QSAXPY_V_M1		4
#define KARGS_QSAXPY_V_M2		5
#define KARGS_QSAXPY_V_MDF		6

#define KARGS_Q_SET_VALUE		0
#define KARGS_Q_SET_Q			1
#define KARGS_Q_SET_M1			2
#define KARGS_Q_SET_M2			3
#define KARGS_Q_SET_MDF			4

#define KARGS_V_SET_VALUE		0
#define KARGS_V_SET_V			1
#define KARGS_V_SET_N1			2
#define KARGS_V_SET_N2			3
#define KARGS_V_SET_NDF			4

#define KARGS_MULMAT_Y			0
#define KARGS_MULMAT_X			1
#define KARGS_MULMAT_LWORK		2
#define KARGS_MULMAT_R2			3
#define KARGS_MULMAT_R3			4
#define KARGS_MULMAT_H			5
#define KARGS_MULMAT_M1			6
#define KARGS_MULMAT_M2			7
#define KARGS_MULMAT_MDF		8

#define KARGS_FORMB_B			0
#define KARGS_FORMB_P2			1
#define KARGS_FORMB_L2			2
#define KARGS_FORMB_PSI			3
#define KARGS_FORMB_L3			4
#define KARGS_FORMB_LWORK		5
#define KARGS_FORMB_R2			6
#define KARGS_FORMB_R3			7
#define KARGS_FORMB_H			8
#define KARGS_FORMB_M1			9
#define KARGS_FORMB_M2			10
#define KARGS_FORMB_MDF			11
#define KARGS_FORMB_NDF			12

#define KARGS_FORMG_G			0
#define KARGS_FORMG_F			1
#define KARGS_FORMG_P1			2
#define KARGS_FORMG_L1			3
#define KARGS_FORMG_LWORK		4
#define KARGS_FORMG_R1			5
#define KARGS_FORMG_H			6
#define KARGS_FORMG_N1			7
#define KARGS_FORMG_N2			8
#define KARGS_FORMG_NDF			9
#define KARGS_FORMG_MDF			10
#define KARGS_FORMG_NDF2		11

#define KARGS_COPYU_U			0
#define KARGS_COPYU_F			1
#define KARGS_COPYU_G			2
#define KARGS_COPYU_N1			3
#define KARGS_COPYU_N2			4
#define KARGS_COPYU_NDF			5
#define KARGS_COPYU_NDF2		6

#define KARGS_SOLVE_SUB1_P1		0
#define KARGS_SOLVE_SUB1_P2		1
#define KARGS_SOLVE_SUB1_U		2
#define KARGS_SOLVE_SUB1_P3		3
#define KARGS_SOLVE_SUB1_L1		4
#define KARGS_SOLVE_SUB1_L2		5
#define KARGS_SOLVE_SUB1_LWORK	6
#define KARGS_SOLVE_SUB1_R1		7
#define KARGS_SOLVE_SUB1_R2		8
#define KARGS_SOLVE_SUB1_H		9
#define KARGS_SOLVE_SUB1_TOL	10
#define KARGS_SOLVE_SUB1_FIRST	11
#define KARGS_SOLVE_SUB1_M1		12
#define KARGS_SOLVE_SUB1_M2		13
#define KARGS_SOLVE_SUB1_NDF	14
#define KARGS_SOLVE_SUB1_MDF	15

#define ARGS_SOLVE_SUB3_PSI		0
#define ARGS_SOLVE_SUB3_P3		1
#define ARGS_SOLVE_SUB3_L3		2
#define ARGS_SOLVE_SUB3_LWORK	3
#define ARGS_SOLVE_SUB3_R3		4
#define ARGS_SOLVE_SUB3_EPS		5
#define ARGS_SOLVE_SUB3_H		6
#define ARGS_SOLVE_SUB3_N1		7
#define ARGS_SOLVE_SUB3_N2		8
#define ARGS_SOLVE_SUB3_NDF		9
#define ARGS_SOLVE_SUB3_MDF		10

#define ARGS_UPDATE_L12_L1		0
#define ARGS_UPDATE_L12_L2		1
#define ARGS_UPDATE_L12_U		2
#define ARGS_UPDATE_L12_P1		3
#define ARGS_UPDATE_L12_P2		4
#define ARGS_UPDATE_L12_P3		5
#define ARGS_UPDATE_L12_LWORK	6
#define ARGS_UPDATE_L12_R1		7
#define ARGS_UPDATE_L12_R2		8
#define ARGS_UPDATE_L12_H		9
#define ARGS_UPDATE_L12_M1		10
#define ARGS_UPDATE_L12_M2		11
#define ARGS_UPDATE_L12_NDF		12
#define ARGS_UPDATE_L12_MDF		13

#define ARGS_UPDATE_L3_L3		0
#define ARGS_UPDATE_L3_P3		1
#define ARGS_UPDATE_L3_PSI		2
#define ARGS_UPDATE_L3_LWORK	3
#define ARGS_UPDATE_L3_R3		4
#define ARGS_UPDATE_L3_H		5
#define ARGS_UPDATE_L3_N1		6
#define ARGS_UPDATE_L3_N2		7
#define ARGS_UPDATE_L3_NDF		8
#define ARGS_UPDATE_L3_MDF		9

#define ARGS_COMPUTE_OBJ_VALUE	0
#define ARGS_COMPUTE_OBJ_U		1
#define ARGS_COMPUTE_OBJ_F		2
#define ARGS_COMPUTE_OBJ_P1		3
#define ARGS_COMPUTE_OBJ_P2		4
#define ARGS_COMPUTE_OBJ_P3		5
#define ARGS_COMPUTE_OBJ_PSI	6
#define ARGS_COMPUTE_OBJ_L1		7
#define ARGS_COMPUTE_OBJ_L2		8
#define ARGS_COMPUTE_OBJ_L3		9
#define ARGS_COMPUTE_OBJ_LWORK	10
#define ARGS_COMPUTE_OBJ_EPS	11
#define ARGS_COMPUTE_OBJ_R1		12
#define ARGS_COMPUTE_OBJ_R2		13
#define ARGS_COMPUTE_OBJ_R3		14
#define ARGS_COMPUTE_OBJ_H		15
#define ARGS_COMPUTE_OBJ_N1		16
#define ARGS_COMPUTE_OBJ_N2		17
#define ARGS_COMPUTE_OBJ_NDF	18
#define ARGS_COMPUTE_OBJ_MDF	19


using namespace auglag1gpu;

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

AuglagL1gpu::TmpBuffers::TmpBuffers(cl::Context &context, int n1, int n2, int ndf, int ndf2, int m1, int m2, int mdf, double h, int sumBufferSize) {
	this->context = context; 
	this->n1 = n1;
	this->n2 = n2;
	this->ndf = ndf;
	this->m1 = m1;
	this->m2 = m2;
	this->mdf = mdf;
	this->sumBufferSize = sumBufferSize;
	this->ldf2BufferSize = (n2-2)*ndf2;
	this->h = h;
}
	
Q& AuglagL1gpu::TmpBuffers::getQ(int i) {
	while((int) qs.size() <= i)
		qs.push_back(Q(context, m1, m2, mdf, h));
	
	return qs.at(i);
}

V& AuglagL1gpu::TmpBuffers::getV(int i) {
	while((int) vs.size() <= i)
		vs.push_back(V(context, n1, n2, ndf, h));
	
	return vs.at(i);
}

cl::Buffer AuglagL1gpu::TmpBuffers::getSumBuffer(int i, cl_int *err) {
	while((int) sumBuffers.size() <= i)
		sumBuffers.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sumBufferSize*sizeof(double), 0, err));
	
	return sumBuffers.at(i);
}

int AuglagL1gpu::TmpBuffers::getSumBufferSize() const {
	return sumBufferSize;
}

cl::Buffer AuglagL1gpu::TmpBuffers::getLdf2Buffer(int i, cl_int *err) {
	while((int) ldf2Buffers.size() <= i)
		ldf2Buffers.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, ldf2BufferSize*sizeof(double), 0, err));
	
	return ldf2Buffers.at(i);
}

int AuglagL1gpu::TmpBuffers::getLdf2BufferSize() const {
	return ldf2BufferSize;
}


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

AuglagL1gpu::AuglagL1gpu(cl::Context &context, cl::Device &device, cl_int *err) {

	cl::Program::Sources sources;
	sources.push_back(cl::Program::Sources::value_type((const char*) auglagL1gpu_cl, auglagL1gpu_cl_len));
	
	std::vector<cl::Device> devices;
	devices.push_back(device);
	
	cl::Program program(context, sources, err);
	
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::AuglagL1gpu: Cannot create program object. " << CLErrorMessage(*err) << std::endl;
		return;
	}
	
	*err = program.build(devices, "");
	
#if !FULL_DEBUG
	if(*err != CL_SUCCESS) {
#endif
		std::string log;
		program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &log);
		if(*err != CL_SUCCESS) {
			std::cerr << "<error> AuglagL1gpu::AuglagL1gpu: OpenCL compiler output:" << std::endl;
			std::cerr << log;
			std::cerr << "<error> AuglagL1gpu::AuglagL1gpu: Cannot build program-object. " << CLErrorMessage(*err) << std::endl;
			return;
		} else {
			std::cout << "<debug> AuglagL1gpu::AuglagL1gpu: OpenCL compiler output:" << std::endl;
			std::cout << log;
		}
#if !FULL_DEBUG
	}
#endif
	
	k_gsum = cl::Kernel(program, "gsum", err);
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::AuglagL1gpu: Cannot create k_gsum kernel object. " << CLErrorMessage(*err) << std::endl;
		return;
	}
	
	k_qdot = cl::Kernel(program, "qdot", err);
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::AuglagL1gpu: Cannot create k_qdot kernel object. " << CLErrorMessage(*err) << std::endl;
		return;
	}
	
	k_qdot_s = cl::Kernel(program, "qdot_s", err);
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::AuglagL1gpu: Cannot create k_qdot_s kernel object. " << CLErrorMessage(*err) << std::endl;
		return;
	}
	
	k_qsaxpy = cl::Kernel(program, "qsaxpy", err);
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::AuglagL1gpu: Cannot create k_qsaxpy kernel object. " << CLErrorMessage(*err) << std::endl;
		return;
	}
	
	k_q_set = cl::Kernel(program, "q_set", err);
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::AuglagL1gpu: Cannot create k_q_set kernel object. " << CLErrorMessage(*err) << std::endl;
		return;
	}
	
	k_v_set = cl::Kernel(program, "v_set", err);
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::AuglagL1gpu: Cannot create k_v_set kernel object. " << CLErrorMessage(*err) << std::endl;
		return;
	}
	
	k_mulMat = cl::Kernel(program, "mulMat", err);
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::AuglagL1gpu: Cannot create k_mulMat kernel object. " << CLErrorMessage(*err) << std::endl;
		return;
	}
	
	k_formB = cl::Kernel(program, "formB", err);
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::AuglagL1gpu: Cannot create k_formB kernel object. " << CLErrorMessage(*err) << std::endl;
		return;
	}
	
	k_formG = cl::Kernel(program, "formG", err);
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::AuglagL1gpu: Cannot create k_formG kernel object. " << CLErrorMessage(*err) << std::endl;
		return;
	}
	
	k_copyU = cl::Kernel(program, "copyU", err);
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::AuglagL1gpu: Cannot create k_copyU kernel object. " << CLErrorMessage(*err) << std::endl;
		return;
	}
	
	k_solve_sub1 = cl::Kernel(program, "solve_sub1", err);
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::AuglagL1gpu: Cannot create k_solve_sub1 kernel object. " << CLErrorMessage(*err) << std::endl;
		return;
	}
	
	k_solve_sub3 = cl::Kernel(program, "solve_sub3", err);
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::AuglagL1gpu: Cannot create k_solve_sub3 kernel object. " << CLErrorMessage(*err) << std::endl;
		return;
	}
	
	k_update_l12 = cl::Kernel(program, "update_l12", err);
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::AuglagL1gpu: Cannot create k_update_l12 kernel object. " << CLErrorMessage(*err) << std::endl;
		return;
	}
	
	k_update_l3 = cl::Kernel(program, "update_l3", err);
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::AuglagL1gpu: Cannot create k_update_l3 kernel object. " << CLErrorMessage(*err) << std::endl;
		return;
	}
	
	k_compute_obj = cl::Kernel(program, "compute_obj", err);
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::AuglagL1gpu: Cannot create k_compute_obj kernel object. " << CLErrorMessage(*err) << std::endl;
		return;
	}
	
	*err = CL_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

double AuglagL1gpu::gsum(cl::CommandQueue &queue, cl::Buffer &sum, int size, cl_int *err) {
#if FULL_DEBUG
	std::cout << "<debug> AuglagL1gpu::gsum: Enqueueing gsum()..." << std::endl;
#endif
	
	const int niceWgGroups = 16;
	const int maxLocalSize = 512;
	const int minLocalSize = 64;
	const int niceElemPerWg = 2;
	
	int partExp = ceil(log2(size));
	int partCount = 1;
	int localSize = std::min(maxLocalSize, 1 << partExp);
	
	while(partCount < niceWgGroups && minLocalSize <= (1 << (partExp-1))) {
		partExp--;
		partCount = DIVCEIL(size, 1 << partExp);
		localSize = std::min(maxLocalSize, 1 << partExp);
	}
	while((1 << partExp)/niceElemPerWg < localSize && minLocalSize <= localSize/2)
		localSize /= 2;
	
	k_gsum.setArg(KARGS_GSUM_SUM, sum);
	k_gsum.setArg(KARGS_GSUM_LWORK, cl::__local(localSize*sizeof(double)));
	k_gsum.setArg(KARGS_GSUM_SIZE, size);
	k_gsum.setArg(KARGS_GSUM_PART_EXP, partExp);
	k_gsum.setArg(KARGS_GSUM_JUMP_EXP, 0);
		
	*err = queue.enqueueNDRangeKernel(k_gsum, cl::NDRange(0, 0, 0), cl::NDRange(partCount*localSize, 1, 1), cl::NDRange(localSize, 1, 1), 0, 0);
	
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::gsum: Cannot enqueue kernel (step 1/2). " << CLErrorMessage(*err) << std::endl;
		return NAN;
	}
	
	if(1 < partCount) {
	
		int oldPartCount = partCount;
		int oldPartExp = partExp;
			
		// Only one work group
		partExp = ceil(log2(size));
		partCount = 1;
		localSize = std::min(maxLocalSize, DIVCEIL(oldPartCount, 32)*32);
		
		k_gsum.setArg(KARGS_GSUM_PART_EXP, partExp);
		
		// new jumpExp = old partExp
		k_gsum.setArg(KARGS_GSUM_JUMP_EXP, oldPartExp);
		
		*err = queue.enqueueNDRangeKernel(k_gsum, cl::NDRange(0, 0, 0), cl::NDRange(localSize, 1, 1), cl::NDRange(localSize, 1, 1), 0, 0);
		
		if(*err != CL_SUCCESS) {
			std::cerr << "<error> AuglagL1gpu::gsum: Cannot enqueue kernel (step 2/2). " << CLErrorMessage(*err) << std::endl;
			return NAN;
		}
	}
	
	double result;
	*err = queue.enqueueReadBuffer(sum, true, 0, sizeof(double), &result, 0, 0);
	
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::gsum: Cannot read from device memory. " << CLErrorMessage(*err) << std::endl;
		return NAN;
	}
	
	*err = CL_SUCCESS;
	return result;
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

double AuglagL1gpu::qdot(cl::CommandQueue &queue, TmpBuffers &tmp, Q &x1, Q &x2, int localSize, cl_int *err) {
#if FULL_DEBUG
	std::cout << "<debug> AuglagL1gpu::qdot: Enqueueing qdot()..." << std::endl;
#endif
	
	int wgCount0 = DIVCEIL(x1.getM1(), localSize);
	int globalSize0 = DIVCEIL(x1.getM1(), localSize)*localSize;
	
	int wgCount1 = std::min(4*x1.getM1(), tmp.getSumBufferSize() / wgCount0);

	cl::Buffer dot = tmp.getSumBuffer(0, err);
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::qdot: Cannot allocate temprary buffer. " << CLErrorMessage(*err) << std::endl;
		return NAN;
	}
	
	
	k_qdot.setArg(KARGS_QDOT_DOT, dot);
	k_qdot.setArg(KARGS_QDOT_X1, x1.getHandle());
	k_qdot.setArg(KARGS_QDOT_X2, x2.getHandle());
	k_qdot.setArg(KARGS_QDOT_LWORK, cl::__local(localSize*sizeof(double)));
	k_qdot.setArg(KARGS_QDOT_M1, x1.getM1());
	k_qdot.setArg(KARGS_QDOT_M2, x1.getM2());
	k_qdot.setArg(KARGS_QDOT_MDF, x1.getMdf());

	*err = queue.enqueueNDRangeKernel(k_qdot, cl::NDRange(0, 0, 0), cl::NDRange(globalSize0, wgCount1, 1), cl::NDRange(localSize, 1, 1), 0, 0);

	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::qdot: Cannot enqueue kernel. " << CLErrorMessage(*err) << std::endl;
		return NAN;
	}
	
	*err = CL_SUCCESS;
	
	return gsum(queue, dot, wgCount0*wgCount1, err);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

double AuglagL1gpu::qdot_s(cl::CommandQueue &queue, TmpBuffers &tmp, Q &x, int localSize, cl_int *err) {
#if FULL_DEBUG
	std::cout << "<debug> AuglagL1gpu::qdot_s: Enqueueing qdot_s()..." << std::endl;
#endif
	
	int wgCount0 = DIVCEIL(x.getM1(), localSize);
	int globalSize0 = DIVCEIL(x.getM1(), localSize)*localSize;
	
	int wgCount1 = std::min(4*x.getM2(), tmp.getSumBufferSize() / wgCount0);
	
	cl::Buffer dot = tmp.getSumBuffer(0, err);
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::qdot_s: Cannot allocate temprary buffer. " << CLErrorMessage(*err) << std::endl;
		return NAN;
	}
	
	k_qdot_s.setArg(KARGS_QDOT_S_DOT, dot);
	k_qdot_s.setArg(KARGS_QDOT_S_X, x.getHandle());
	k_qdot_s.setArg(KARGS_QDOT_S_LWORK, cl::__local(localSize*sizeof(double)));
	k_qdot_s.setArg(KARGS_QDOT_S_M1, x.getM1());
	k_qdot_s.setArg(KARGS_QDOT_S_M2, x.getM2());
	k_qdot_s.setArg(KARGS_QDOT_S_MDF, x.getMdf());
	
	*err = queue.enqueueNDRangeKernel(k_qdot_s, cl::NDRange(0, 0, 0), cl::NDRange(globalSize0, wgCount1, 1), cl::NDRange(localSize, 1, 1), 0, 0);

	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::qdot_s: Cannot enqueue kernel. " << CLErrorMessage(*err) << std::endl;
		return NAN;
	}
	
	*err = CL_SUCCESS;
	
	return gsum(queue, dot, wgCount0*wgCount1, err);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

void AuglagL1gpu::qsaxpy(cl::CommandQueue &queue, Q &y, Q &x1, double a, Q &x2, int localSize, cl_int *err) {
#if FULL_DEBUG
	std::cout << "<debug> AuglagL1gpu::qsaxpy: Enqueueing qsaxpy()..." << std::endl;
#endif
	
	const int niceGlobalSize = 15*16*512;
	
	int globalSize0 = DIVCEIL(y.getM1(), localSize)*localSize;
	int wgCount1 = std::min(4*y.getM2(), niceGlobalSize/globalSize0);
	

	k_qsaxpy.setArg(KARGS_QSAXPY_Y, y.getHandle());
	k_qsaxpy.setArg(KARGS_QSAXPY_X1, x1.getHandle());
	k_qsaxpy.setArg(KARGS_QSAXPY_A, a);
	k_qsaxpy.setArg(KARGS_QSAXPY_X2, x2.getHandle());
	k_qsaxpy.setArg(KARGS_QSAXPY_M1, y.getM1());
	k_qsaxpy.setArg(KARGS_QSAXPY_M2, y.getM2());
	k_qsaxpy.setArg(KARGS_QSAXPY_MDF, y.getMdf());
	
	*err = queue.enqueueNDRangeKernel(k_qsaxpy, cl::NDRange(0, 0, 0), cl::NDRange(globalSize0, wgCount1, 1), cl::NDRange(localSize, 1, 1), 0, 0);

	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::qsaxpy: Cannot enqueue kernel. " << CLErrorMessage(*err) << std::endl;
		return;
	}
	
	*err = CL_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

void AuglagL1gpu::q_set(cl::CommandQueue &queue, double value, Q &q, int localSize, cl_int *err) {
#if FULL_DEBUG
	std::cout << "<debug> AuglagL1gpu::q_set: Enqueueing q_set()..." << std::endl;
#endif
	
	const int niceGlobalSize = 15*16*512;
	
	int globalSize0 = DIVCEIL(q.getM1(), localSize)*localSize;
	int wgCount1 = std::min(4*q.getM2(), niceGlobalSize/globalSize0);
	
	k_q_set.setArg(KARGS_Q_SET_VALUE, value);
	k_q_set.setArg(KARGS_Q_SET_Q, q.getHandle());
	k_q_set.setArg(KARGS_Q_SET_M1, q.getM1());
	k_q_set.setArg(KARGS_Q_SET_M2, q.getM2());
	k_q_set.setArg(KARGS_Q_SET_MDF, q.getMdf());
	
	*err = queue.enqueueNDRangeKernel(k_q_set, cl::NDRange(0, 0, 0), cl::NDRange(globalSize0, wgCount1, 1), cl::NDRange(localSize, 1, 1), 0, 0);

	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::q_set: Cannot enqueue kernel. " << CLErrorMessage(*err) << std::endl;
		return;
	}
	
	*err = CL_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

void AuglagL1gpu::v_set(cl::CommandQueue &queue, double value, V &v, int localSize, cl_int *err) {
#if FULL_DEBUG
	std::cout << "<debug> AuglagL1gpu::v_set: Enqueueing v_set()..." << std::endl;
#endif
	
	const int niceGlobalSize = 15*16*512;
	
	int globalSize0 = DIVCEIL(v.getN1(), localSize)*localSize;
	int wgCount1 = std::min(v.getN2(), niceGlobalSize/globalSize0);
	
	k_v_set.setArg(KARGS_V_SET_VALUE, value);
	k_v_set.setArg(KARGS_V_SET_V, v.getHandle());
	k_v_set.setArg(KARGS_V_SET_N1, v.getN1());
	k_v_set.setArg(KARGS_V_SET_N2, v.getN2());
	k_v_set.setArg(KARGS_V_SET_NDF, v.getNdf());
	
	*err = queue.enqueueNDRangeKernel(k_v_set, cl::NDRange(0, 0, 0), cl::NDRange(globalSize0, wgCount1, 1), cl::NDRange(localSize, 1, 1), 0, 0);

	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::v_set: Cannot enqueue kernel. " << CLErrorMessage(*err) << std::endl;
		return;
	}
	
	*err = CL_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

void AuglagL1gpu::mulMat(cl::CommandQueue &queue, Q &y, Q &x, double r2, double r3, int localSize, cl_int *err) {
#if FULL_DEBUG
	std::cout << "<debug> AuglagL1gpu::mulMat: Enqueueing mulMat()..." << std::endl;
#endif

	k_mulMat.setArg(KARGS_MULMAT_Y, y.getHandle());
	k_mulMat.setArg(KARGS_MULMAT_X, x.getHandle());
//	k_mulMat.setArg(KARGS_MULMAT_LWORK, cl::__local(localSize*sizeof(double)));
	k_mulMat.setArg(KARGS_MULMAT_LWORK, cl::__local(0));
	k_mulMat.setArg(KARGS_MULMAT_R2, r2);
	k_mulMat.setArg(KARGS_MULMAT_R3, r3);
	k_mulMat.setArg(KARGS_MULMAT_H, y.getH());
	k_mulMat.setArg(KARGS_MULMAT_M1, y.getM1());
	k_mulMat.setArg(KARGS_MULMAT_M2, y.getM2());
	k_mulMat.setArg(KARGS_MULMAT_MDF, y.getMdf());
	
	int globalSize = DIVCEIL(y.getM1(), localSize)*localSize;
	
	*err = queue.enqueueNDRangeKernel(k_mulMat, cl::NDRange(0, 0, 0), cl::NDRange(globalSize, y.getM2(), 4), cl::NDRange(localSize, 1, 1), 0, 0);

	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::mulMat: Cannot enqueue kernel. " << CLErrorMessage(*err) << std::endl;
		return;
	}

	*err = CL_SUCCESS;
	return;
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

void AuglagL1gpu::formB(cl::CommandQueue& queue, Q& b, Q& p2, Q& l2, V& psi, V& l3, double r2, double r3, int localSize, cl_int *err) {
#if FULL_DEBUG
	std::cout << "<debug> AuglagL1gpu::formB: Enqueueing formB()..." << std::endl;
#endif

	k_formB.setArg(KARGS_FORMB_B, b.getHandle());
	k_formB.setArg(KARGS_FORMB_P2, p2.getHandle());
	k_formB.setArg(KARGS_FORMB_L2, l2.getHandle());
	k_formB.setArg(KARGS_FORMB_PSI, psi.getHandle());
	k_formB.setArg(KARGS_FORMB_L3, l3.getHandle());
	k_formB.setArg(KARGS_FORMB_LWORK, cl::__local(2*localSize*sizeof(double)));
	k_formB.setArg(KARGS_FORMB_R2, r2);
	k_formB.setArg(KARGS_FORMB_R3, r3);
	k_formB.setArg(KARGS_FORMB_H, b.getH());
	k_formB.setArg(KARGS_FORMB_M1, b.getM1());
	k_formB.setArg(KARGS_FORMB_M2, b.getM2());
	k_formB.setArg(KARGS_FORMB_MDF, b.getMdf());
	k_formB.setArg(KARGS_FORMB_NDF, psi.getNdf());
	
	int globalSize = DIVCEIL(b.getM1(), localSize)*localSize;
	
	*err = queue.enqueueNDRangeKernel(k_formB, cl::NDRange(0, 0, 0), cl::NDRange(globalSize, b.getM2(), 4), cl::NDRange(localSize, 1, 1), 0, 0);

	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::formB: Cannot enqueue kernel. " << CLErrorMessage(*err) << std::endl;
		return;
	}

	*err = CL_SUCCESS;

}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

void AuglagL1gpu::formG(cl::CommandQueue &queue, cl::Buffer &g, V &f, Q &p1, Q &l1, double r1, int ndf2, int localSize, cl_int *err) {
#if FULL_DEBUG
	std::cout << "<debug> AuglagL1gpu::formG: Enqueueing formG()..." << std::endl;
#endif


	k_formG.setArg(KARGS_FORMG_G, g);
	k_formG.setArg(KARGS_FORMG_F, f.getHandle());
	k_formG.setArg(KARGS_FORMG_P1, p1.getHandle());
	k_formG.setArg(KARGS_FORMG_L1, l1.getHandle());
	k_formG.setArg(KARGS_FORMG_LWORK, cl::__local(2*localSize*sizeof(double)));
	k_formG.setArg(KARGS_FORMG_R1, r1);
	k_formG.setArg(KARGS_FORMG_H, f.getH());
	k_formG.setArg(KARGS_FORMG_N1, f.getN1());
	k_formG.setArg(KARGS_FORMG_N2, f.getN2());
	k_formG.setArg(KARGS_FORMG_NDF, f.getNdf());
	k_formG.setArg(KARGS_FORMG_MDF, p1.getMdf());
	k_formG.setArg(KARGS_FORMG_NDF2, ndf2);
	
	int globalSize = DIVCEIL(f.getN1()-2, localSize)*localSize;
	
	*err = queue.enqueueNDRangeKernel(k_formG, cl::NDRange(0, 0, 0), cl::NDRange(globalSize, f.getN2()-2, 1), cl::NDRange(localSize, 1, 1), 0, 0);

	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::formG: Cannot enqueue kernel. " << CLErrorMessage(*err) << std::endl;
		return;
	}

	*err = CL_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

void AuglagL1gpu::copyU(cl::CommandQueue &queue, V &u, V &f, cl::Buffer &g, int ndf2, int localSize, cl_int *err) {
#if FULL_DEBUG
	std::cout << "<debug> AuglagL1gpu::copyU: Enqueueing copyU()..." << std::endl;
#endif
	
	k_copyU.setArg(KARGS_COPYU_U, u.getHandle());
	k_copyU.setArg(KARGS_COPYU_F, f.getHandle());
	k_copyU.setArg(KARGS_COPYU_G, g);
	k_copyU.setArg(KARGS_COPYU_N1, u.getN1());
	k_copyU.setArg(KARGS_COPYU_N2, u.getN2());
	k_copyU.setArg(KARGS_COPYU_NDF, u.getNdf());
	k_copyU.setArg(KARGS_COPYU_NDF2, ndf2);
	
	int globalSize = DIVCEIL(u.getN1(), localSize)*localSize;
	
	*err = queue.enqueueNDRangeKernel(k_copyU, cl::NDRange(0, 0, 0), cl::NDRange(globalSize, u.getN2(), 1), cl::NDRange(localSize, 1, 1), 0, 0);

	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::copyU: Cannot enqueue kernel. " << CLErrorMessage(*err) << std::endl;
		return;
	}

	*err = CL_SUCCESS;
	
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

void AuglagL1gpu::solve_sub1(
	cl::CommandQueue &queue, Q &p1, Q &p2, V &u, Q &p3, Q &l1, Q &l2, 
	double r1, double r2, double tol, bool first, int localSize, cl_int *err) {
	
#if FULL_DEBUG
	std::cout << "<debug> AuglagL1gpu::solve_sub1: Enqueueing solve_sub1()..." << std::endl;
#endif
	
	k_solve_sub1.setArg(KARGS_SOLVE_SUB1_P1, p1.getHandle());
	k_solve_sub1.setArg(KARGS_SOLVE_SUB1_P2, p2.getHandle());
	k_solve_sub1.setArg(KARGS_SOLVE_SUB1_U, u.getHandle());
	k_solve_sub1.setArg(KARGS_SOLVE_SUB1_P3, p3.getHandle());
	k_solve_sub1.setArg(KARGS_SOLVE_SUB1_L1, l1.getHandle());
	k_solve_sub1.setArg(KARGS_SOLVE_SUB1_L2, l2.getHandle());
	k_solve_sub1.setArg(KARGS_SOLVE_SUB1_LWORK, cl::__local(localSize*sizeof(double)));
	k_solve_sub1.setArg(KARGS_SOLVE_SUB1_R1, r1);
	k_solve_sub1.setArg(KARGS_SOLVE_SUB1_R2, r2);
	k_solve_sub1.setArg(KARGS_SOLVE_SUB1_H, p1.getH());
	k_solve_sub1.setArg(KARGS_SOLVE_SUB1_TOL, tol);
	k_solve_sub1.setArg(KARGS_SOLVE_SUB1_FIRST, first ? 1 : 0);
	k_solve_sub1.setArg(KARGS_SOLVE_SUB1_M1, p1.getM1());
	k_solve_sub1.setArg(KARGS_SOLVE_SUB1_M2, p1.getM2());
	k_solve_sub1.setArg(KARGS_SOLVE_SUB1_NDF, u.getNdf());
	k_solve_sub1.setArg(KARGS_SOLVE_SUB1_MDF, p1.getMdf());
	
	// HACK
	localSize = 64;
	
	int globalSize = DIVCEIL(p1.getM1(), localSize)*localSize;
	
	*err = queue.enqueueNDRangeKernel(k_solve_sub1, cl::NDRange(0, 0, 0), cl::NDRange(globalSize, p1.getM2(), 2), cl::NDRange(localSize, 1, 1), 0, 0);
	
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::solve_sub1: Cannot enqueue kernel. " << CLErrorMessage(*err) << std::endl;
		return;
	}
	
	*err = CL_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

void AuglagL1gpu::solve_sub2(cl::CommandQueue &queue, TmpBuffers &tmp, Q &p3, V &psi, Q &p2, Q &l2, V &l3,
		double r2, double r3, double tol, int max_iter, int localSize, cl_int *err) {
	
	// HACK
	localSize = 256;
	
	double resid, normb;
	double alpha, beta, rho, rhop;
	
	Q b = tmp.getQ(0);
	Q r = tmp.getQ(1);
	Q p = tmp.getQ(2);
	Q Ap = tmp.getQ(3);
	
	// Form b
	formB(queue, b, p2, l2, psi, l3, r2, r3, localSize, err);
	if(*err != CL_SUCCESS)
		return;

	// normb = || b ||
	normb = sqrt(qdot_s(queue, tmp, b, localSize, err));
	if(*err != CL_SUCCESS)
		return;

	// Ap = A*x
	mulMat(queue, Ap, p3, r2, r3, localSize, err);
	if(*err != CL_SUCCESS)
		return;
	
	// r = b - Ap = b - A*x
	qsaxpy(queue, r, b, -1.0, Ap, localSize, err);
	if(*err != CL_SUCCESS)
		return;

	if (normb == 0.0)
		normb = 1;

	// rhop = < r_0 , r_0 >
	rhop = qdot_s(queue, tmp, r, localSize, err);
	if(*err != CL_SUCCESS)
		return;
	
	// || r || = || b - A*x || < tol ???
	if ((resid = sqrt(rhop) / normb) <= tol) {
		return;
	}
	
	// p = r
	devCopy(queue, p, r, err);
	if(*err != CL_SUCCESS)
		return;

	for (int i = 1; i <= max_iter; i++) {
		// rho = < r_i , r_i >
		rho = qdot_s(queue, tmp, r, localSize, err);
		if(*err != CL_SUCCESS)
			return;
		
		// Ap = A*p
		mulMat(queue, Ap, p, r2, r3, localSize, err);
		if(*err != CL_SUCCESS)
			return;

		// alpha = rho / < p , Ap > = < r , r > / < p , A*p >
		alpha = rho / qdot(queue, tmp, p, Ap, localSize, err);
		if(*err != CL_SUCCESS)
			return;
	
		// x = x + alpha * p
		qsaxpy(queue, p3, p3, alpha, p, localSize, err);
		if(*err != CL_SUCCESS)
			return;
		
		// r_{i+1 = r_i - alpha * Ab
		qsaxpy(queue, r, r, -alpha, Ap, localSize, err);
		if(*err != CL_SUCCESS)
			return;
		
		// rhop = < r_{i+1} , r_{i+1} >
		rhop = qdot_s(queue, tmp, r, localSize, err);
		if(*err != CL_SUCCESS)
			return;
		
		// || r || < tol ???
		if ((resid = sqrt(rhop) / normb) <= tol) {
			return;
		}

		// beta = < r_{i+1} , r_{i+1} > / < r_i , r_i >
		beta = rhop / rho;
		
		// p = r + beta * p
		qsaxpy(queue, p, r, beta, p, localSize, err);
		if(*err != CL_SUCCESS)
			return;

#if DEBUG
		if(i % 100 == 0)
			std::cout << "<debug> CG: " << i << " iterations, |r| = " << resid << std::endl;
#endif

	}

}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

void AuglagL1gpu::solve_sub3(cl::CommandQueue &queue, V &psi, Q &p3, V &l3, double r3, double eps, int localSize, cl_int *err) {
#if FULL_DEBUG
	std::cout << "<debug> AuglagL1gpu::solve_sub3: Enqueueing solve_sub3()..." << std::endl;
#endif

	k_solve_sub3.setArg(ARGS_SOLVE_SUB3_PSI, psi.getHandle());
	k_solve_sub3.setArg(ARGS_SOLVE_SUB3_P3, p3.getHandle());
	k_solve_sub3.setArg(ARGS_SOLVE_SUB3_L3, l3.getHandle());
	k_solve_sub3.setArg(ARGS_SOLVE_SUB3_LWORK, cl::__local(localSize*sizeof(double)));
	k_solve_sub3.setArg(ARGS_SOLVE_SUB3_R3, r3);
	k_solve_sub3.setArg(ARGS_SOLVE_SUB3_EPS, eps);
	k_solve_sub3.setArg(ARGS_SOLVE_SUB3_H, psi.getH());
	k_solve_sub3.setArg(ARGS_SOLVE_SUB3_N1, psi.getN1());
	k_solve_sub3.setArg(ARGS_SOLVE_SUB3_N2, psi.getN2());
	k_solve_sub3.setArg(ARGS_SOLVE_SUB3_NDF, psi.getNdf());
	k_solve_sub3.setArg(ARGS_SOLVE_SUB3_MDF, p3.getMdf());
	
	int globalSize = DIVCEIL(psi.getN1(), localSize)*localSize;
	
	*err = queue.enqueueNDRangeKernel(k_solve_sub3, cl::NDRange(0, 0, 0), cl::NDRange(globalSize, psi.getN2(), 1), cl::NDRange(localSize, 1, 1), 0, 0);
	
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::solveSub3: Cannot enqueue kernel. " << CLErrorMessage(*err) << std::endl;
		return;
	}
	
	*err = CL_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

pscrCL::L2SolverContext* AuglagL1gpu::init_sub4(cl::Context &context, cl::Device &device, cl::CommandQueue &queue, int n1, int n2, int ndf2, double h, cl_int *err) {
#if FULL_DEBUG
	std::cout << "<debug> AuglagL1gpu::init_sub4: Initializing pscrCL solver..." << std::endl;
#endif

	double a1Diag[n2-2], a1OffDiag[n2-2], m1Diag[n2-2];
	double a2Diag[n1-2], a2OffDiag[n1-2], m2Diag[n1-2];
	for(int i = 0; i < n2-2; i++) {
		a1Diag[i] = 2.0;
		a1OffDiag[i] = -1.0;
		m1Diag[i] = 1;
	}

	for(int i = 0; i < n1-2; i++) {
		a2Diag[i] = 2.0;
		a2OffDiag[i] = -1.0;
		m2Diag[i] = 1;
	}

	pscrCL::PscrCLMode mode(PSCRCL_PREC_DOUBLE | PSCRCL_NUM_REAL);
	
	std::vector<cl::Context> contexts;
	contexts.push_back(context);
	
	std::vector<cl::Device> oneDevice;
	oneDevice.push_back(device);
			
	std::vector<pscrCL::OptValues> optValues = pscrCL::L2SolverContext::getDefaultValues(oneDevice, n1-2, mode);

	pscrCL::L2SolverContext *solver = 0;
	try {
		solver = new pscrCL::L2SolverContext(
			contexts,
			oneDevice,
			optValues,
			a1Diag,
			a1OffDiag,
			m1Diag,
			0,
			a2Diag,
			a2OffDiag,
			m2Diag,
			0,
			n2-2,
			n1-2,
			n2-2,
			ndf2,
			mode);
		
		std::vector<pscrCL::CommandQueue> queues;
		queues.push_back(queue);
		solver->allocate(queues);
		
		solver->allocateTmp();
	} catch (...) {
		std::cerr << "<error> AuglagL1gpu::init_sub4: Error while initializing the pscrCL solver. " << CLErrorMessage(*err) << std::endl;
		*err = -9999;
		return 0;
	}
	
	*err = CL_SUCCESS;
	
	return solver;
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

void AuglagL1gpu::solve_sub4(cl::CommandQueue &queue, pscrCL::L2SolverContext *solver, TmpBuffers &tmp, V &u, V &f, Q &p1, Q &l1, double r1, int ndf2, int localSize, cl_int *err) {
#if FULL_DEBUG
	std::cout << "<debug> AuglagL1gpu::solve_sub4: Calling pscrCL solver..." << std::endl;
#endif
	
	cl::Buffer g = tmp.getLdf2Buffer(0, err);
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::solve_sub4: Error while allocating g-vector. " << CLErrorMessage(*err) << std::endl;
		return;
	}
	
	formG(queue, g, f, p1, l1, r1, ndf2, localSize, err);
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::solve_sub4: Error while forming g-vector. " << CLErrorMessage(*err) << std::endl;
		return;
	}

	std::vector<pscrCL::CommandQueue> queues;
	std::vector<cl::Buffer> devMemFs;
	
	queues.push_back(queue);
	devMemFs.push_back(g);
	
	double ch = u.getH()*u.getH()/r1;
	
	try {
		solver->run(queues, devMemFs, 1, &ch);
	} catch (...) {
		std::cerr << "<error> AuglagL1gpu::solve_sub4: Error while running the pscrCL solver. " << CLErrorMessage(*err) << std::endl;
		*err = -9999;
		return;
	}
	
	copyU(queue, u, f, g, ndf2, localSize, err);
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::solve_sub4: Error while forming u-vector. " << CLErrorMessage(*err) << std::endl;
	}

}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

void AuglagL1gpu::updateLambdas(cl::CommandQueue &queue, Q &l1, Q &l2, V &l3, V &u, Q &p1, Q &p2, Q &p3, V &psi, double r1, double r2, double r3, int localSize, cl_int *err) {

#if FULL_DEBUG
	std::cout << "<debug> AuglagL1gpu::updateLambdas: Enqueueing update_l12()..." << std::endl;
#endif
	
	k_update_l12.setArg(ARGS_UPDATE_L12_L1, l1.getHandle());
	k_update_l12.setArg(ARGS_UPDATE_L12_L2, l2.getHandle());
	k_update_l12.setArg(ARGS_UPDATE_L12_U, u.getHandle());
	k_update_l12.setArg(ARGS_UPDATE_L12_P1, p1.getHandle());
	k_update_l12.setArg(ARGS_UPDATE_L12_P2, p2.getHandle());
	k_update_l12.setArg(ARGS_UPDATE_L12_P3, p3.getHandle()),
	k_update_l12.setArg(ARGS_UPDATE_L12_LWORK, cl::__local(localSize*sizeof(double)));
	k_update_l12.setArg(ARGS_UPDATE_L12_R1, r1);
	k_update_l12.setArg(ARGS_UPDATE_L12_R2, r2);
	k_update_l12.setArg(ARGS_UPDATE_L12_H, l1.getH());
	k_update_l12.setArg(ARGS_UPDATE_L12_M1, l1.getM1());
	k_update_l12.setArg(ARGS_UPDATE_L12_M2, l1.getM2());
	k_update_l12.setArg(ARGS_UPDATE_L12_NDF, u.getNdf());
	k_update_l12.setArg(ARGS_UPDATE_L12_MDF, l1.getMdf());
	
	int globalSize1 = DIVCEIL(l1.getM1(), localSize)*localSize;
	
	*err = queue.enqueueNDRangeKernel(k_update_l12, cl::NDRange(0, 0, 0), cl::NDRange(globalSize1, l1.getM2(), 2), cl::NDRange(localSize, 1, 1), 0, 0);
	
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::updateLambdas: Cannot enqueue k_update_l12 kernel. " << CLErrorMessage(*err) << std::endl;
		return;
	}

#if FULL_DEBUG
	std::cout << "<debug> AuglagL1gpu::updateLambdas: Enqueueing update_l3()..." << std::endl;
#endif
	
	k_update_l3.setArg(ARGS_UPDATE_L3_L3, l3.getHandle());
	k_update_l3.setArg(ARGS_UPDATE_L3_P3, p3.getHandle());
	k_update_l3.setArg(ARGS_UPDATE_L3_PSI, psi.getHandle());
	k_update_l3.setArg(ARGS_UPDATE_L3_LWORK, cl::__local(localSize*sizeof(double)));
	k_update_l3.setArg(ARGS_UPDATE_L3_R3, r3);
	k_update_l3.setArg(ARGS_UPDATE_L3_H, l3.getH());
	k_update_l3.setArg(ARGS_UPDATE_L3_N1, l3.getN1());
	k_update_l3.setArg(ARGS_UPDATE_L3_N2, l3.getN2());
	k_update_l3.setArg(ARGS_UPDATE_L3_NDF, l3.getNdf());
	k_update_l3.setArg(ARGS_UPDATE_L3_MDF, p3.getMdf());
	
	int globalSize2 = DIVCEIL(l3.getN1(), localSize)*localSize;
	
	*err = queue.enqueueNDRangeKernel(k_update_l3, cl::NDRange(0, 0, 0), cl::NDRange(globalSize2, l3.getN2(), 1), cl::NDRange(localSize, 1, 1), 0, 0);
	
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::updateLambdas: Cannot enqueue k_update_l3 kernel. " << CLErrorMessage(*err) << std::endl;
		return;
	}
	
	*err = CL_SUCCESS;
	
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

double AuglagL1gpu::computeObject(cl::CommandQueue &queue, TmpBuffers &tmp, V &u, V &f, Q &p1, Q &p2, Q &p3, V &psi, Q &l1, Q &l2, V &l3, double eps, double r1, double r2, double r3, int localSize, cl_int *err) {

	cl::Buffer tmpBuffer = tmp.getSumBuffer(0, err);
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::computeObject: Cannot allocate temporary buffer. " << CLErrorMessage(*err) << std::endl;
		return NAN;
	}
	
#if FULL_DEBUG
	std::cout << "<debug> AuglagL1gpu::computeObject: Enqueueing compute_obj()..." << std::endl;
#endif
	
	k_compute_obj.setArg( ARGS_COMPUTE_OBJ_VALUE, tmpBuffer);
	k_compute_obj.setArg( ARGS_COMPUTE_OBJ_U, u.getHandle());
	k_compute_obj.setArg( ARGS_COMPUTE_OBJ_F, f.getHandle());
	k_compute_obj.setArg( ARGS_COMPUTE_OBJ_P1, p1.getHandle());
	k_compute_obj.setArg( ARGS_COMPUTE_OBJ_P2, p2.getHandle());
	k_compute_obj.setArg( ARGS_COMPUTE_OBJ_P3, p3.getHandle());
	k_compute_obj.setArg( ARGS_COMPUTE_OBJ_PSI, psi.getHandle());
	k_compute_obj.setArg( ARGS_COMPUTE_OBJ_L1, l1.getHandle());
	k_compute_obj.setArg( ARGS_COMPUTE_OBJ_L2, l2.getHandle());
	k_compute_obj.setArg( ARGS_COMPUTE_OBJ_L3, l3.getHandle());
	k_compute_obj.setArg( ARGS_COMPUTE_OBJ_LWORK, cl::__local(localSize*sizeof(double)));
	k_compute_obj.setArg( ARGS_COMPUTE_OBJ_EPS, eps);
	k_compute_obj.setArg( ARGS_COMPUTE_OBJ_R1, r1);
	k_compute_obj.setArg( ARGS_COMPUTE_OBJ_R2, r2);
	k_compute_obj.setArg( ARGS_COMPUTE_OBJ_R3, r3);
	k_compute_obj.setArg( ARGS_COMPUTE_OBJ_H, u.getH());
	k_compute_obj.setArg( ARGS_COMPUTE_OBJ_N1, u.getN1());
	k_compute_obj.setArg( ARGS_COMPUTE_OBJ_N2, u.getN1());
	k_compute_obj.setArg( ARGS_COMPUTE_OBJ_NDF, u.getNdf());
	k_compute_obj.setArg( ARGS_COMPUTE_OBJ_MDF, p1.getMdf());
	
	int gn1 = DIVCEIL(p1.getM1(), localSize);
	
	*err = queue.enqueueNDRangeKernel(k_compute_obj, cl::NDRange(0, 0, 0), cl::NDRange(gn1*localSize, p1.getM2(), 1), cl::NDRange(localSize, 1, 1), 0, 0);
	
	if(*err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::computeObject: Cannot enqueue k_compute_obj kernel. " << CLErrorMessage(*err) << std::endl;
		return NAN;
	}
	
	double value = gsum(queue, tmpBuffer, gn1*l1.getM2(), err);
	
	if(*err != CL_SUCCESS) {
		return NAN;
	}
	
	return value;
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

void AuglagL1gpu::solve(cl::Context &context, cl::Device &device, cl::CommandQueue &queue, double *input, double *output, int width, int height, int ldfw, double r1, double r2, double r3, double eps, double h, double delta, int maxIter) {

	cl_int err;
	
	int n1 = width;
	int n2 = height;
	
	int m1 = n1-1;
	int m2 = n2-1;
	
	int ndf = DIVCEIL(n1, 32)*32;
	int mdf = DIVCEIL(m1, 32)*32;
	int ndf2 = DIVCEIL(n1-2, 32)*32;
	
	int localSize = DIVCEIL(n1, 32)*32;
	while(512 < localSize)
		localSize = DIVCEIL(localSize/2, 32)*32;
	
	int tmpSize = DIVCEIL(m1, localSize) * n2;
	
	err = queue.finish();
	if(err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::solve: Error while finalizing command queue. " << CLErrorMessage(err) << std::endl;
		return;
	}
	
	Timer initTimer;
	initTimer.begin();

	pscrCL::L2SolverContext* pscr = init_sub4(context, device, queue, n1, n2, ndf2, h, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::solve: Cannot initialize pscrCL solver. " << CLErrorMessage(err) << std::endl;
		return;
	}
	
	//
	// Allocate devíce memory
	//
	
	V u(context, n1, n2, ndf, h);
	V f(context, n1, n2, ndf, h);

	Q p1(context, m1, m2, mdf, h);
	Q p2(context, m1, m2, mdf, h);
	Q p3(context, m1, m2, mdf, h);
	
	Q l1(context, m1, m2, mdf, h);
	Q l2(context, m1, m2, mdf, h);
	V l3(context, n1, n2, ndf, h);
	
	V psi(context, n1, n2, ndf, h);
	
	AuglagL1gpu::TmpBuffers tmp(context, n1, n2, ndf, ndf2, m1, m2, mdf, h, tmpSize);
	tmp.getLdf2Buffer(0, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::solve: Error while calling tmp.getLdf2Buffer(). " << CLErrorMessage(err) << std::endl;
		delete pscr;
		return;
	}
	tmp.getSumBuffer(0, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::solve: Error while calling tmp.getSumBuffer(). " << CLErrorMessage(err) << std::endl;
		delete pscr;
		return;
	}
	tmp.getQ(3);
	
	//
	// Initialize devíce memory
	//
	
	v_set(queue, 0.0, u, localSize, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::solve: Error while initializing u. " << CLErrorMessage(err) << std::endl;
		delete pscr;
		return;
	}
	
	q_set(queue, 0.0, p1, localSize, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::solve: Error while initializing p1. " << CLErrorMessage(err) << std::endl;
		delete pscr;
		return;
	}
	q_set(queue, 0.0, p2, localSize, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::solve: Error while initializing p2. " << CLErrorMessage(err) << std::endl;
		delete pscr;
		return;
	}
	q_set(queue, 0.0, p3, localSize, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::solve: Error while initializing p3. " << CLErrorMessage(err) << std::endl;
		delete pscr;
		return;
	}
	
	q_set(queue, 0.0, l1, localSize, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::solve: Error while initializing l1. " << CLErrorMessage(err) << std::endl;
		delete pscr;
		return;
	}
	q_set(queue, 0.0, l2, localSize, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::solve: Error while initializing l2. " << CLErrorMessage(err) << std::endl;
		delete pscr;
		return;
	}
	v_set(queue, 0.0, l3, localSize, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::solve: Error while initializing l3. " << CLErrorMessage(err) << std::endl;
		delete pscr;
		return;
	}
	
	v_set(queue, 0.0, psi, localSize, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::solve: Error while initializing psi. " << CLErrorMessage(err) << std::endl;
		delete pscr;
		return;
	}
	
	err = queue.finish();
	if(err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::solve: Error while finalizing command queue. " << CLErrorMessage(err) << std::endl;
		delete pscr;
		return;
	}
	initTimer.end();
	
	Timer fullSolutionTimer;
	fullSolutionTimer.begin();
	
	//
	// Copy noisy image to the device memory
	//
	
	for(int j = 0; j < n2; j++)
		for(int i = 0; i < n1; i++)
			f(i,j) = input[j*ldfw+i];
		
	f.push(queue);
	
	err = queue.finish();
	if(err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::solve: Error while finalizing command queue. " << CLErrorMessage(err) << std::endl;
		delete pscr;
		return;
	}
	
	Timer deviceSolutionTimer;
	deviceSolutionTimer.begin();
	
	//
	// Calculate object function value at f
	//
	double objValueF = computeObject(queue, tmp, f, f, p1, p2, p3, psi, l1, l2, l3, eps, r1, r2, r3, localSize, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::solve: Error while calling computeObject(). " << CLErrorMessage(err) << std::endl;
		delete pscr;
		return;
	}
	
#if DEBUG
	std::cout << "<debug> J(f) = " << objValueF << std::endl;
#endif
	
	//
	// Calculate initial object function value
	//
	double objValueOrg = computeObject(queue, tmp, u, f, p1, p2, p3, psi, l1, l2, l3, eps, r1, r2, r3, localSize, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::solve: Error while calling computeObject(). " << CLErrorMessage(err) << std::endl;
		delete pscr;
		return;
	}
	
#if DEBUG
	std::cout << "<debug> J(u_0) = " << objValueOrg << std::endl;
#endif
	
	//
	// Main iteration
	//
	
	double objValueOld = objValueOrg;
	
	std::vector<std::pair<cl::Event,cl::Event> > sub1Events;
	std::vector<std::pair<cl::Event,cl::Event> > sub2Events;
	std::vector<std::pair<cl::Event,cl::Event> > sub3Events;
	std::vector<std::pair<cl::Event,cl::Event> > sub4Events;
	std::vector<std::pair<cl::Event,cl::Event> > miscEvents;
	
	int k;
	for(k = 1; k <= maxIter; k++) {
		
		cl::Event event0;
		queue.enqueueMarker(&event0);
		
		solve_sub1(queue, p1, p2, u, p3, l1, l2, r1, r2, 1.0E-8, k <= 2, localSize, &err);
		if(err != CL_SUCCESS) {
			std::cerr << "<error> AuglagL1gpu::solve: Error while calling solve_sub1(). " << CLErrorMessage(err) << std::endl;
			delete pscr;
			return;
		}

		cl::Event event1;
		queue.enqueueMarker(&event1);
		
		sub1Events.push_back(std::pair<cl::Event,cl::Event>(event0, event1));
		
		solve_sub2(queue, tmp, p3, psi, p2, l2, l3, r2, r3, 1.0E-3, 30, localSize, &err);
		if(err != CL_SUCCESS) {
			std::cerr << "<error> AuglagL1gpu::solve: Error while calling solve_sub2(). " << CLErrorMessage(err) << std::endl;
			delete pscr;
			return;
		}

		cl::Event event2;
		queue.enqueueMarker(&event2);
		
		sub2Events.push_back(std::pair<cl::Event,cl::Event>(event1, event2));
		
		solve_sub3(queue, psi, p3, l3, r3, eps, localSize, &err);
		if(err != CL_SUCCESS) {
			std::cerr << "<error> AuglagL1gpu::solve: Error while calling solve_sub3(). " << CLErrorMessage(err) << std::endl;
			delete pscr;
			return;
		}

		cl::Event event3;
		queue.enqueueMarker(&event3);
		
		sub3Events.push_back(std::pair<cl::Event,cl::Event>(event2, event3));
		
		solve_sub4(queue, pscr, tmp, u, f, p1, l1, r1, ndf2, localSize, &err);
		if(err != CL_SUCCESS) {
			std::cerr << "<error> AuglagL1gpu::solve: Error while calling solve_sub4(). " << CLErrorMessage(err) << std::endl;
			delete pscr;
			return;
		}
		
		cl::Event event4;
		queue.enqueueMarker(&event4);
		
		sub4Events.push_back(std::pair<cl::Event,cl::Event>(event3, event4));
		
		updateLambdas(queue, l1, l2, l3, u, p1, p2, p3, psi, r1, r2, r3, localSize, &err);
		if(err != CL_SUCCESS) {
			std::cerr << "<error> AuglagL1gpu::solve: Error while calling updateLambdas(). " << CLErrorMessage(err) << std::endl;
			delete pscr;
			return;
		}

		double objValueNew = computeObject(queue, tmp, u, f, p1, p2, p3, psi, l1, l2, l3, eps, r1, r2, r3, localSize, &err);
		if(err != CL_SUCCESS) {
			std::cerr << "<error> AuglagL1gpu::solve: Error while calling computeObject(). " << CLErrorMessage(err) << std::endl;
			delete pscr;
			return;
		}
		
		cl::Event event5;
		queue.enqueueMarker(&event5);
		
		miscEvents.push_back(std::pair<cl::Event,cl::Event>(event4, event5));
		
#if FULL_DEBUG
		u.pull(queue);
		p1.pull(queue);
		p2.pull(queue);
		p3.pull(queue);
		psi.pull(queue);
		
		double resid1 = 0.0;
		double ugrad = 0.0;
		double resid2 = 0.0;
		
		for(int j = 0; j < m2; j++) {
			for(int i = 0; i < m1; i++) {
				ugrad += dot(u.grad(i,j,V::odd)) + dot(u.grad(i,j,V::even));
				resid1 += dot(u.grad(i,j,V::odd) - p1(i,j,Q::odd));
				resid1 += dot(u.grad(i,j,V::even) - p1(i,j,Q::even));
				resid2 += dot(p2(i,j,Q::odd)-p3(i,j,Q::odd));
				resid2 += dot(p2(i,j,Q::even)-p3(i,j,Q::even));
			}
		}
		
		std::cout << "<debug> resid1 = " << 0.5*sqrt(resid1)/std::max(sqrt(ugrad), norm2(p1)) << std::endl;
		std::cout << "<debug> resid2 = " << 0.5*sqrt(resid2)/std::max(norm2(p2), norm2(p3)) << std::endl;
		
		double resid3 = 0.0;
		double p3div = 0.0;
		
		for(int j = 0; j < n2; j++) {
			for(int i = 0; i < n1; i++) {
				p3div += p3.div(i,j);
				resid3 += (p3.div(i,j)-psi(i,j))*(p3.div(i,j)-psi(i,j));
			}
		}
		
		std::cout << "<debug> resid3 = " << 0.5*sqrt(resid3)/std::max(sqrt(p3div), norm2(psi)) << std::endl;
	
#endif
		
#if FULL_DEBUG
		std::cout << "<debug> J(u_i) = " << objValueNew << std::endl;
#endif
		
		double ratio = ::fabs((objValueNew-objValueOld)/objValueOld);
		
#if DEBUG
		std::cout << "<debug> Iter #" << k << "; |last-current|/|last| = " << ratio << std::endl;
#endif

		if(ratio < delta && 5 < k) {
#if DEBUG
			std::cout << "<debug> Success." << std::endl;
#endif
			break;
		}
		
		objValueOld = objValueNew;
		
	}
	
	err = queue.finish();
	if(err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::solve: Error while finalizing command queue. " << CLErrorMessage(err) << std::endl;
		delete pscr;
		return;
	}
	deviceSolutionTimer.end();
	
	//
	// Copy denoised image from the device memory
	//
	u.pull(queue);
	
	for(int j = 0; j < n2; j++)
		for(int i = 0; i < n1; i++)
			output[j*ldfw+i] = u(i,j);
		
	err = queue.finish();
	if(err != CL_SUCCESS) {
		std::cerr << "<error> AuglagL1gpu::solve: Error while finalizing command queue. " << CLErrorMessage(err) << std::endl;
		delete pscr;
		return;
	}
	fullSolutionTimer.end();
	
	delete pscr;
	
	std::cout << "<debug> Initialization time:             " << initTimer.getTime() << " s" << std::endl;
	std::cout << "<debug> Full solution time:              " << fullSolutionTimer.getTime() << " s" << std::endl;
	std::cout << "<debug> Device solution time:            " << deviceSolutionTimer.getTime() << " s" << std::endl;
	std::cout << "<debug> Average step time:               " << deviceSolutionTimer.getTime()/k << " s" << std::endl;
	
	double sub1TotalTime = 0.0;
	std::vector<std::pair<cl::Event,cl::Event> >::iterator sub1Iter;
	for(sub1Iter = sub1Events.begin(); sub1Iter != sub1Events.end(); sub1Iter++) {
		cl_ulong begin;
		sub1Iter->first.getProfilingInfo(CL_PROFILING_COMMAND_START, &begin);
		cl_ulong end;
		sub1Iter->second.getProfilingInfo(CL_PROFILING_COMMAND_START, &end);
		sub1TotalTime += 1.0e-9 * (end-begin);
	}
	
	double sub2TotalTime = 0.0;
	std::vector<std::pair<cl::Event,cl::Event> >::iterator sub2Iter;
	for(sub2Iter = sub2Events.begin(); sub2Iter != sub2Events.end(); sub2Iter++) {
		cl_ulong begin;
		sub2Iter->first.getProfilingInfo(CL_PROFILING_COMMAND_START, &begin);
		cl_ulong end;
		sub2Iter->second.getProfilingInfo(CL_PROFILING_COMMAND_START, &end);
		sub2TotalTime += 1.0e-9 * (end-begin);
	}
	
	double sub3TotalTime = 0.0;
	std::vector<std::pair<cl::Event,cl::Event> >::iterator sub3Iter;
	for(sub3Iter = sub3Events.begin(); sub3Iter != sub3Events.end(); sub3Iter++) {
		cl_ulong begin;
		sub3Iter->first.getProfilingInfo(CL_PROFILING_COMMAND_START, &begin);
		cl_ulong end;
		sub3Iter->second.getProfilingInfo(CL_PROFILING_COMMAND_START, &end);
		sub3TotalTime += 1.0e-9 * (end-begin);
	}
	
	double sub4TotalTime = 0.0;
	std::vector<std::pair<cl::Event,cl::Event> >::iterator sub4Iter;
	for(sub4Iter = sub4Events.begin(); sub4Iter != sub4Events.end(); sub4Iter++) {
		cl_ulong begin;
		sub4Iter->first.getProfilingInfo(CL_PROFILING_COMMAND_START, &begin);
		cl_ulong end;
		sub4Iter->second.getProfilingInfo(CL_PROFILING_COMMAND_START, &end);
		sub4TotalTime += 1.0e-9 * (end-begin);
	}
	
	double miscTotalTime = 0.0;
	std::vector<std::pair<cl::Event,cl::Event> >::iterator miscIter;
	for(miscIter = miscEvents.begin(); miscIter != miscEvents.end(); miscIter++) {
		cl_ulong begin;
		miscIter->first.getProfilingInfo(CL_PROFILING_COMMAND_START, &begin);
		cl_ulong end;
		miscIter->second.getProfilingInfo(CL_PROFILING_COMMAND_START, &end);
		miscTotalTime += 1.0e-9 * (end-begin);
	}
	
	std::cout << "<debug> Subproblem #1 total time:        " << sub1TotalTime << " s" << std::endl;
	std::cout << "<debug> Subproblem #2 total time:        " << sub2TotalTime << " s" << std::endl;
	std::cout << "<debug> Subproblem #3 total time:        " << sub3TotalTime << " s" << std::endl;
	std::cout << "<debug> Subproblem #4 total time:        " << sub4TotalTime << " s" << std::endl;
	std::cout << "<debug> Misc total time:                 " << miscTotalTime << " s" << std::endl;

	
	std::cout << "<debug> Subproblem #1 average step time: " << sub1TotalTime/k << " s" << std::endl;
	std::cout << "<debug> Subproblem #2 average step time: " << sub2TotalTime/k << " s" << std::endl;
	std::cout << "<debug> Subproblem #3 average step time: " << sub3TotalTime/k << " s" << std::endl;
	std::cout << "<debug> Subproblem #4 average step time: " << sub4TotalTime/k << " s" << std::endl;
	std::cout << "<debug> Misc average step time:          " << miscTotalTime/k << " s" << std::endl;
}
