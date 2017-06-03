#ifndef AUGLAGL1GPU_H_
#define AUGLAGL1GPU_H_

#include "cl.hpp"
#include <vector>

#include "L2SolverContext.h"

#include "Q.h"
#include "V.h"

namespace auglag1gpu {

class AuglagL1gpu {
public:
    class TmpBuffers {
    public:
        TmpBuffers(cl::Context &context, int n1, int n2, int ndf, int ndf2, int m1, int m2, int mdf, double h, int sumBufferSize);
        Q& getQ(int i);
        V& getV(int i);
        cl::Buffer getSumBuffer(int i, cl_int *err);
        int getSumBufferSize() const;
        cl::Buffer getLdf2Buffer(int i, cl_int *err);
        int getLdf2BufferSize() const;
    private:
        cl::Context context;
        int n1;
        int n2;
        int ndf;
        int m1;
        int m2;
        int mdf;
        int sumBufferSize;
        int ldf2BufferSize;
        double h;
        std::vector<Q> qs;
        std::vector<V> vs;
        std::vector<cl::Buffer> sumBuffers;
        std::vector<cl::Buffer> ldf2Buffers;
    };

    AuglagL1gpu(cl::Context &context, cl::Device &device, cl_int *err);
	
	void solve(cl::Context &context, cl::Device &device, cl::CommandQueue &queue, double *input, double *output, int width, int height, int ldfw, double r1, double r2, double r3, double eps, double h, double delta, int maxIter);

    void solve_sub1(cl::CommandQueue &queue, Q &p1, Q &p2, V &u, Q &p3, Q &l1, Q &l2,
                    double r1, double r2, double tol, bool first, int localSize, cl_int *err);

    void solve_sub2(cl::CommandQueue &queue, TmpBuffers &tmp, Q &p3, V &psi, Q &p2, Q &l2, V &l3,
                    double r2, double r3, double tol, int max_iter, int localSize, cl_int *err);

    void solve_sub3(cl::CommandQueue &queue, V &psi, Q &p3, V &l3, double r3, double eps, int localSize, cl_int *err);

    pscrCL::L2SolverContext* init_sub4(cl::Context &context, cl::Device &device, cl::CommandQueue &queue, int n1, int n2, int ndf2, double h, cl_int *err);

    void solve_sub4(cl::CommandQueue &queue, pscrCL::L2SolverContext *solver, TmpBuffers &tmp, V &u, V &f, Q &p1, Q &l1, double r1, int ndf2, int localSize, cl_int *err);

	void updateLambdas(cl::CommandQueue &queue, Q &l1, Q &l2, V &l3, V &u, Q &p1, Q &p2, Q &p3, V &psi, double r1, double r2, double r3, int localSize, cl_int *err);
	
	double computeObject(cl::CommandQueue &queue, TmpBuffers &tmp, V &u, V &f, Q &p1, Q &p2, Q &p3, V &psi, Q &l1, Q &l2, V &l3, double eps, double r1, double r2, double r3, int localSize, cl_int *err);

    double gsum(cl::CommandQueue &queue, cl::Buffer &sum, int size, cl_int *err);

    double qdot(cl::CommandQueue &queue, TmpBuffers &tmp, Q &x1, Q &x2, int localSize, cl_int *err);
    double qdot_s(cl::CommandQueue &queue, TmpBuffers &tmp, Q &x, int localSize, cl_int *err);

    void qsaxpy(cl::CommandQueue &queue, Q &y, Q &x1, double a, Q &x2, int localSize, cl_int *err);

    void q_set(cl::CommandQueue &queue, double value, Q &q, int localSize, cl_int *err);
    void v_set(cl::CommandQueue &queue, double value, V &v, int localSize, cl_int *err);

    void mulMat(cl::CommandQueue &queue, Q &y, Q &x, double r2, double r3, int localSize, cl_int *err);
    void formB(cl::CommandQueue &queue, Q &b, Q &p2, Q &l2, V &psi, V &l3, double r2, double r3, int localSize, cl_int *err);

    void formG(cl::CommandQueue &queue, cl::Buffer &g, V &f, Q &p1, Q &l1, double r1, int ndf2, int localSize, cl_int *err);
	void copyU(cl::CommandQueue &queue, V &u, V &f, cl::Buffer &g, int ndf2, int localSize, cl_int *err);

private:
    cl::Kernel k_gsum;

    cl::Kernel k_qdot;
    cl::Kernel k_qdot_s;
    cl::Kernel k_qsaxpy;

    cl::Kernel k_q_set;
    cl::Kernel k_v_set;

    cl::Kernel k_mulMat;
    cl::Kernel k_formB;
	
    cl::Kernel k_formG;
	cl::Kernel k_copyU;

    cl::Kernel k_solve_sub1;
    cl::Kernel k_solve_sub3;
	
	cl::Kernel k_update_l12;
	cl::Kernel k_update_l3;
	
	cl::Kernel k_compute_obj;
};

}

#endif

