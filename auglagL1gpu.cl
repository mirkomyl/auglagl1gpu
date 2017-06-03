#define NEWTON_DIV 1

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double _var;
typedef double2 _var2;

#define INF (1.0/0.0)

#if NEWTON_DIV
union invType {
	double f;
	unsigned long long i;
	struct {
		unsigned int e1;
		unsigned int e2;
	} ii;
};
double inv(union invType b) {
	union invType md, x;
	
	md.i = (b.i & 0x000fffffffffffff) | 0xbfe0000000000000;
	
	x.f = (48.0/17.0) + (32.0/17.0) * md.f;
	
	x.f = x.f + x.f * (1.0 + md.f * x.f);
	x.f = x.f + x.f * (1.0 + md.f * x.f);
	x.f = x.f + x.f * (1.0 + md.f * x.f);
	x.f = x.f + x.f * (1.0 + md.f * x.f);

	x.ii.e2 += 0x3fe00000 - (b.ii.e2 & 0x7ff00000);
	x.i |= b.i & 0x8000000000000000;
	
	return x.f;
}
#define INV(b) inv((union invType)(b))
#else
#define INV(b) (1.0/(b))
#endif

_var pow2(_var x) { return x*x; }

_var pow3(_var x) { return x*x*x; }

_var pow5(_var x) { return x*x*x*x*x; }

_var pow3_2(_var x) { return sqrt(pow3(x)); }

_var pow5_2(_var x) { return sqrt(pow5(x)); }

_var sgn(_var x) { return x < 0 ? -1 : 1; }

_var norm_var2(_var2 x) {
	return sqrt(x.s0*x.s0 + x.s1*x.s1);
}

_var dot_s_var2(_var2 x) {
	return x.s0*x.s0 + x.s1*x.s1;
}

_var dot_var2(_var2 x1, _var2 x2) {
	return x1.s0*x2.s0 + x1.s1*x2.s1;
}

#define ODD__TRI 0
#define EVEN_TRI 1

_var2 grad(__global _var *u, __local _var *lwork, int i, int j, int comp, int n1, int n2, int ndf, _var h) {
	const int local_id = get_local_id(0);
	const int local_size = get_local_size(0);
	
	// +--------
	// |******/
	// |****/
	// |**/
	// |/
	if(comp == ODD__TRI) {
		_var ul; 
		if(i < n1) {
			ul = u[(j+1)*ndf+i];
			lwork[local_id] = ul;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if(i+1 < n1) {
			_var ur;
			if(local_id < local_size-1)
				ur = lwork[local_id+1];
			else
				ur = u[(j+1)*ndf+i+1];
				
			_var ll = u[j*ndf+i];
			
			return (_var2)(ur-ul, ul-ll)*INV(h);
		}
	}
	
	//       /|
	//     /**|
	//   /****|
	// /******|
	//--------+
	if(comp == EVEN_TRI) {
		_var ll; 
		if(i < n1) {
			ll = u[j*ndf+i];
			lwork[local_id] = ll;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if(i+1 < n1) {
			_var lr;
			if(local_id < local_size-1)
				lr = lwork[local_id+1];
			else
				lr = u[j*ndf+i+1];

			_var ur = u[(j+1)*ndf+i+1];
			
			return (_var2)(lr-ll, ur-lr)*INV(h);
		}
	}
	
	return 0.0;
}

_var div(__global _var *q, __local _var *lwork, int i, int j, int m1, int m2, int mdf, _var h) {
	const int local_id = get_local_id(0);
	const int local_size = get_local_size(0);

	//(j <= 0 || n2-1 <= j)
	if(j <= 0 || m2 <= j)
		return 0.0;
	
	// |  /|  /|
	// |/>>|/  |
	// +---+---+
	// |  /|  /|
	// |/  |/  |
	_var2 e1;
	if(0 <= i-1 && i-1 < m1) {
		e1.x = q[(m2+j)*mdf+i-1];
		lwork[local_id] = e1.x;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	_var e2, e3;
	if(0 <= i-1 && i < m1) {
		// |  /|  /|
		// |/^^|/  |
		// +^^^+---+
		// |  /|  /|
		// |/  |/  |
		e1.y = q[(3*m2+j)*mdf+i-1];
	
	
		// |  /|^^/|
		// |/  |/  |
		// +---+---+
		// |  /|  /|
		// |/  |/  |
		e2 = q[(2*m2+j)*mdf+i];
	
		// |  /|  /|
		// |/  |/>>|
		// +---+---+
		// |  /|  /|
		// |/  |/  |
		if(local_id < local_size-1)
			e3 = lwork[local_id+1];
		else
			e3 = q[(m2+j)*mdf+i];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// |  /|  /|
	// |/  |/  |
	// +---+---+
	// |>>/|  /|
	// |/  |/  |
	_var e4;
	if(0 <= i-1 && i-1 < m1) {
		e4 = q[(j-1)*mdf+i-1];
		lwork[local_id] = e4;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	_var e5; _var2 e6;
	if(0 <= i-1 && i < m1) {
	
		// |  /|  /|
		// |/  |/  |
		// +---+---+
		// |  /|  /|
		// |/^^|/  |
		e5 = q[(3*m2+j-1)*mdf+i-1];
		
		// |  /|  /|
		// |/  |/  |
		// +---+---+
		// |  /|>>/|
		// |/  |/  |
		if(local_id < local_size-1)
			e6.x = lwork[local_id+1];
		else
			e6.x = q[(j-1)*mdf+i];
			
		// |  /|  /|
		// |/  |/  |
		// +---+---+
		// |  /|^^/|
		// |/  |/  |
		e6.y = q[(2*m2+j-1)*mdf+i];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	//(0 < i || i < n1-1)
	if(0 < i && i < m1)
		return - (e1.x - e1.y - e2 - e3 + e4 + e5 - e6.x + e6.y) * INV(2.0*h);

	return 0.0;
}

#define LWG_SUM_THRESHOLD 8

_var lsum(__local _var *lwork, int local_id, int size) {
	if(size < 1)
		return 0.0;

	while(LWG_SUM_THRESHOLD < size) {
		int size2 = size/2;
		size = size2 + (size & 0x1);
		if(local_id < size2)
			lwork[local_id] += lwork[size+local_id];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
#if 1 < LWG_SUM_THRESHOLD
	if(local_id == 0) {
		_var sum = 0.0;
		for(int i = 0; i < size; i++)
			sum += lwork[i];
		lwork[0] = sum;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
#endif
	
	return lwork[0];
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

// [1 ... 2^jump_exp ... 2 ... 2^jump_exp ... 3 ... 2^jump_exp ...] => 1 + 2 + 3 + ...
__kernel void gsum(__global _var *sum, __local _var *lwork, int size, int part_exp, int jump_exp) {
	const int local_id = get_local_id(0);
	const int local_size = get_local_size(0);
	const int gid0 = get_group_id(0);

	int size2 = 1 << part_exp;
	__global _var *part = sum + gid0*size2;
	int part_size = min(size2, size-gid0*size2);
	
	_var y = 0.0;

	for(int i = local_id << jump_exp; i < part_size; i += local_size << jump_exp)
		y += part[i];
		
	lwork[local_id] = y;
	barrier(CLK_LOCAL_MEM_FENCE);
	y = lsum(lwork, local_id, local_size);
	
	if(local_id == 0)
		part[0] = y;
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

__kernel void qdot(__global _var *dot, __global _var *x1, __global _var *x2, __local _var *lwork, int m1, int m2, int mdf) {
	const int local_id = get_local_id(0);
	const int global_id = get_global_id(0);
	const int local_size = get_local_size(0);
	const int gid0 = get_group_id(0);
	const int gid1 = get_group_id(1);
	const int ng0 = get_num_groups(0);
	const int ng1 = get_num_groups(1);
	
	_var y = 0.0;
	
	if(global_id < m1) {
		for(int i = gid1*mdf+global_id; i < 4*m2*mdf; i += ng1*mdf) {
			y += x1[i] * x2[i];
		}
	}

	int local_sum_size = min(local_size, m1-gid0*local_size);
	if(local_id < local_sum_size)
		lwork[local_id] = y;
	barrier(CLK_LOCAL_MEM_FENCE);
	y = lsum(lwork, local_id, local_sum_size);
	
	if(local_id == 0)
		dot[gid1*ng0+gid0] = y;
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

__kernel void qdot_s(__global _var *dot, __global _var *x, __local _var *lwork, int m1, int m2, int mdf) {
	const int local_id = get_local_id(0);
	const int global_id = get_global_id(0);
	const int local_size = get_local_size(0);
	const int gid0 = get_group_id(0);
	const int gid1 = get_group_id(1);
	const int ng0 = get_num_groups(0);
	const int ng1 = get_num_groups(1);
	
	_var y = 0.0;
	
	if(global_id < m1) {
		for(int i = gid1*mdf+global_id; i < 4*m2*mdf; i += ng1*mdf) {
			double xx = x[i];
			y += xx*xx;
		}
	}

	int local_sum_size = min(local_size, m1-gid0*local_size);
	if(local_id < local_sum_size)
		lwork[local_id] = y;
	barrier(CLK_LOCAL_MEM_FENCE);
	y = lsum(lwork, local_id, local_sum_size);
	
	if(local_id == 0)
		dot[gid1*ng0+gid0] = y;
}


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

__kernel void qsaxpy(__global _var *y, __global _var *x1, _var a, __global _var *x2, int m1, int m2, int mdf) {
	const int global_id = get_global_id(0);
	const int gid1 = get_group_id(1);
	const int ng1 = get_num_groups(1);

	if(global_id < m1) {
		for(int i = gid1*mdf+global_id; i < 4*m2*mdf; i += ng1*mdf) {
			y[i] = x1[i] + a*x2[i];
		}
	}
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

__kernel void v_set(_var value, __global _var *v, int n1, int n2, int ndf) {
	const int global_id = get_global_id(0);
	const int global_size = get_global_size(0);
	const int gid1 = get_group_id(1);
	const int ng1 = get_num_groups(1);

	for(int j = gid1; j < n2; j += ng1)
		for(int i = global_id; i < ndf; i += global_size)
			v[j*ndf + i] = i < n1 ? value : 0.0;
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

__kernel void q_set(_var value, __global _var *q, int m1, int m2, int mdf) {
	const int global_id = get_global_id(0);
	const int global_size = get_global_size(0);
	const int gid1 = get_group_id(1);
	const int ng1 = get_num_groups(1);

	for(int j = gid1; j < 4*m2; j += ng1) 
		for(int i = global_id; i < mdf; i += global_size)
			q[j*mdf + i] = i < m1 ? value : 0.0;
		
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

// | 1 -1  .  .|
// |-1  2 -1  .|
// | . -1  2 -1| 
// | .  . -1  1|
_var mulMat_ul_m1_p2_m1(const __global _var *x, __local _var *lwork, int m1) {
	const int local_id = get_local_id(0);
	const int local_size = get_local_size(0);
	const int global_id = get_global_id(0);

	_var yy = 0.0;
	
	if(global_id < m1) {
		if(0 < global_id)
			yy -= x[global_id-1];
			
		if(global_id == 0 || global_id == m1-1)
			yy += x[global_id];
		else
			yy += 2.0*x[global_id];
		if(global_id < m1-1)
			yy -= x[global_id+1];
	}
	
	return yy;
	
// 	_var yy = 0.0;
// 
// 	_var middle;
// 	if(global_id < m1) {
// 		middle = x[global_id];
// 		lwork[local_id] = middle;
// 	}
// 	
// 	barrier(CLK_LOCAL_MEM_FENCE);
// 	
// 	if(m1 <= global_id)
// 		return 0.0;
// 	
// 	if(0 < global_id) {
// 		_var left;
// 		if(local_id == 0)
// 			left = x[global_id - 1];
// 		else
// 			left = lwork[local_id - 1];
// 			
// 		yy -= left;
// 	}
// 	
// 	if(global_id == 0 || global_id == m1-1)
// 		yy += middle;
// 	else
// 		yy += 2.0*middle;
// 		
// 	if(global_id < m1-1) {
// 		_var right;
// 		if(local_id == local_size-1)
// 			right = x[global_id + 1];
// 		else
// 			right = lwork[local_id + 1];
// 			
// 		yy -= right;
// 	}
// 	
// 	return yy;
}

// | . -1  .  .|
// | .  1 -1  .|
// | .  .  1 -1| 
// | .  .  .  1|
_var mulMat_ur_00_p1_m1(const __global _var *x, __local _var *lwork, int m1) {
	const int local_id = get_local_id(0);
	const int local_size = get_local_size(0);
	const int global_id = get_global_id(0);
	
	_var yy = 0.0;
	
	if(0 < global_id && global_id < m1)
		yy += x[global_id];
	
	if(global_id < m1-1)
		yy -= x[global_id + 1];
	
	return yy;
	
// 	_var yy = 0.0;
// 
// 	_var middle;
// 	if(global_id < m1) {
// 		middle = x[global_id];
// 		lwork[local_id] = middle;
// 	}
// 	
// 	barrier(CLK_LOCAL_MEM_FENCE);
// 	
// 	if(m1 <= global_id)
// 		return 0.0;
// 	
// 	if(global_id < m1-1) {
// 		_var right;
// 		if(local_id == local_size-1)
// 			right = x[global_id + 1];
// 		else
// 			right = lwork[local_id + 1];
// 			
// 		yy -= right;
// 	}
// 	
// 	if(0 < global_id)
// 		yy += middle;
// 	
// 	return yy;
}

// | 1  .  .  .|
// |-1  1  .  .|
// | . -1  1  .| 
// | .  . -1  .|
_var mulMat_ur_m1_p1_00(const __global _var *x, __local _var *lwork, int m1) {
	const int local_id = get_local_id(0);
	const int local_size = get_local_size(0);
	const int global_id = get_global_id(0);

	_var yy = 0.0;
	
	if(0 < global_id && global_id < m1)
		yy -= x[global_id-1];
	
	if(global_id < m1-1)
		yy += x[global_id];
		
	return yy;
	
// 	_var yy = 0.0;
// 
// 	_var middle;
// 	if(global_id < m1-1) {
// 		middle = x[global_id];
// 		lwork[local_id] = middle;
// 	}
// 	
// 	barrier(CLK_LOCAL_MEM_FENCE);
// 	
// 	if(m1 <= global_id)
// 		return 0.0;
// 	
// 	if(0 < global_id) {
// 		_var left;
// 		if(local_id == 0)
// 			left = x[global_id - 1];
// 		else
// 			left = lwork[local_id - 1];
// 			
// 		yy -= left;
// 	}
// 	
// 	if(global_id < m1-1)
// 		yy += middle;
// 		
// 	return yy;
}

// | 1 -1  .  .|
// | .  1 -1  .|
// | .  .  1 -1| 
// | .  .  .  .|
_var mulMat_ll_00_p1_m1(const __global _var *x, __local _var *lwork, int m1) {
	const int local_id = get_local_id(0);
	const int local_size = get_local_size(0);
	const int global_id = get_global_id(0);
	
	_var yy = 0.0;
	
	if(global_id < m1-1) {
		yy += x[global_id];
		yy -= x[global_id+1];
	}
	
	return yy;

// 	_var yy = 0.0;
// 
// 	_var middle;
// 	if(global_id < m1) {
// 		middle = x[global_id];
// 		lwork[local_id] = middle;
// 	}
// 	
// 	barrier(CLK_LOCAL_MEM_FENCE);
// 	
// 	// Skip last row
// 	if(m1-1 <= global_id)
// 		return 0.0;
// 	
// 	_var right;
// 	if(local_id == local_size-1)
// 		right = x[global_id + 1];
// 	else
// 		right = lwork[local_id + 1];
// 		
// 	yy -= right;
// 	
// 	yy += middle;
// 
// 	return yy;
}

// | .  .  .  .|
// |-1  1  .  .|
// | . -1  1  .| 
// | .  . -1  1|
_var mulMat_ll_m1_p1_00(const __global _var *x, __local _var *lwork, int m1) {
	const int local_id = get_local_id(0);
	const int local_size = get_local_size(0);
	const int global_id = get_global_id(0);
	
	_var yy = 0.0;
	
	if(0 < global_id && global_id < m1) {
		yy -= x[global_id-1];
		yy += x[global_id];
	}
	
	return yy;

// 	_var yy = 0.0;
// 
// 	_var middle;
// 	if(global_id < m1) {
// 		middle = x[global_id];
// 		lwork[local_id] = middle;
// 	}
// 	
// 	barrier(CLK_LOCAL_MEM_FENCE);
// 	
// 	// Skip first row
// 	if(global_id == 0 || m1 <= global_id)
// 		return 0.0;
// 	
// 	_var left;
// 	if(local_id == 0)
// 		left = x[global_id - 1];
// 	else
// 		left = lwork[local_id - 1];
// 		
// 	yy -= left;
// 	
// 	yy += middle;
// 		
// 	return yy;
}

__kernel void mulMat(__global _var *y, const __global _var *x, __local _var *lwork, _var r2, _var r3, _var h, int m1, int m2, int mdf) {
	const int local_id = get_local_id(0);
	const int global_id = get_global_id(0);
	const int local_size = get_local_size(0);
	const int gid0 = get_group_id(0);
	const int gid1 = get_group_id(1); // y-diretion
	const int gid2 = get_group_id(2); 
	
	if(m1 <= gid0*local_size)
		return;
	
	_var yy = 0.0;
	
	if(gid2 == 0) {
	
		if(gid1 < m2-1) {
	
			// odd, x1
			yy += mulMat_ul_m1_p2_m1(x + gid1*mdf, lwork, m1);
// 			barrier(CLK_LOCAL_MEM_FENCE);
			
			// even, x1
			yy += mulMat_ul_m1_p2_m1(x + (m2 + gid1 + 1)*mdf, lwork, m1);
// 			barrier(CLK_LOCAL_MEM_FENCE);
			
			// odd, x2
			yy -= mulMat_ur_00_p1_m1(x + (2*m2 + gid1)*mdf, lwork, m1);
// 			barrier(CLK_LOCAL_MEM_FENCE);
			yy += mulMat_ur_00_p1_m1(x + (2*m2 + gid1 + 1)*mdf, lwork, m1);
// 			barrier(CLK_LOCAL_MEM_FENCE);
			
			// even, x2
			yy += mulMat_ur_m1_p1_00(x + (3*m2 + gid1)*mdf, lwork, m1);
// 			barrier(CLK_LOCAL_MEM_FENCE);
			yy -= mulMat_ur_m1_p1_00(x + (3*m2 + gid1 + 1)*mdf, lwork, m1);
// 			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}
	
	if(gid2 == 1) {
		
		if(0 < gid1) {
			
			// odd, x1
			yy += mulMat_ul_m1_p2_m1(x + (gid1 - 1)*mdf, lwork, m1);
// 			barrier(CLK_LOCAL_MEM_FENCE);
			
			// even, x1
			yy += mulMat_ul_m1_p2_m1(x + (m2 + gid1)*mdf, lwork, m1);
// 			barrier(CLK_LOCAL_MEM_FENCE);
			
			// odd, x2
			yy -= mulMat_ur_00_p1_m1(x + (2*m2 + gid1 - 1)*mdf, lwork, m1);
// 			barrier(CLK_LOCAL_MEM_FENCE);
			yy += mulMat_ur_00_p1_m1(x + (2*m2 + gid1)*mdf, lwork, m1);
// 			barrier(CLK_LOCAL_MEM_FENCE);
			
			// even, x2
			yy += mulMat_ur_m1_p1_00(x + (3*m2 + gid1 - 1)*mdf, lwork, m1);
// 			barrier(CLK_LOCAL_MEM_FENCE);
			yy -= mulMat_ur_m1_p1_00(x + (3*m2 + gid1)*mdf, lwork, m1);
// 			barrier(CLK_LOCAL_MEM_FENCE);
		}
		
	}
	
	if(gid2 == 2) {
	
		// odd, x1
		if(0 < gid1)
			yy += mulMat_ll_m1_p1_00(x + (gid1 - 1)*mdf, lwork, m1);
// 		barrier(CLK_LOCAL_MEM_FENCE);
		
		if(gid1 < m2-1)
			yy -= mulMat_ll_m1_p1_00(x + gid1*mdf, lwork, m1);
// 		barrier(CLK_LOCAL_MEM_FENCE);
		
		// even, x1
		if(0 < gid1)
			yy += mulMat_ll_m1_p1_00(x + (m2 + gid1)*mdf, lwork, m1);
// 		barrier(CLK_LOCAL_MEM_FENCE);
		if(gid1 < m2-1)
			yy -= mulMat_ll_m1_p1_00(x + (m2 + gid1 + 1)*mdf, lwork, m1);
// 		barrier(CLK_LOCAL_MEM_FENCE);
		
		// odd, x2
		if(0 < global_id) {
			if(0 < gid1)
				yy -= x[(2*m2 + gid1 - 1)*mdf + global_id];
				
			if(gid1 == 0 || gid1 == m2-1)
				yy += x[(2*m2 + gid1)*mdf + global_id];
			else
				yy += 2.0*x[(2*m2 + gid1)*mdf + global_id];
			
			if(gid1 < m2-1)
				yy -= x[(2*m2 + gid1 + 1)*mdf + global_id];
		}
		
		// even, x2
		if(0 < global_id) {
			if(0 < gid1)
				yy -= x[(3*m2 + gid1 - 1)*mdf + global_id - 1];
				
			if(gid1 == 0 || gid1 == m2-1)
				yy += x[(3*m2 + gid1)*mdf + global_id - 1];
			else
				yy += 2.0*x[(3*m2 + gid1)*mdf + global_id - 1];
			
			if(gid1 < m2-1)
				yy -= x[(3*m2 + gid1 + 1)*mdf + global_id - 1];
		}
	}
	
	if(gid2 == 3) {
	
		// odd, x1
		if(0 < gid1)
			yy -= mulMat_ll_00_p1_m1(x + (gid1 - 1)*mdf, lwork, m1);
// 		barrier(CLK_LOCAL_MEM_FENCE);
		
		if(gid1 < m2-1)
			yy += mulMat_ll_00_p1_m1(x + gid1*mdf, lwork, m1);
// 		barrier(CLK_LOCAL_MEM_FENCE);

		// even, x1
		if(0 < gid1)
			yy -= mulMat_ll_00_p1_m1(x + (m2 + gid1)*mdf, lwork, m1);
// 		barrier(CLK_LOCAL_MEM_FENCE);
		if(gid1 < m2-1)
			yy += mulMat_ll_00_p1_m1(x + (m2 + gid1 + 1)*mdf, lwork, m1);
// 		barrier(CLK_LOCAL_MEM_FENCE);
		
		// odd, x2
		if(global_id < m1-1) {
			if(0 < gid1)
				yy -= x[(2*m2 + gid1 - 1)*mdf + global_id + 1];
				
			if(gid1 == 0 || gid1 == m2-1)
				yy += x[(2*m2 + gid1)*mdf + global_id + 1];
			else
				yy += 2.0*x[(2*m2 + gid1)*mdf + global_id + 1];
			
			if(gid1 < m2-1)
				yy -= x[(2*m2 + gid1 + 1)*mdf + global_id + 1];
		}
		
		// even, x2
		if(global_id < m1-1) {
			if(0 < gid1)
				yy -= x[(3*m2 + gid1 - 1)*mdf + global_id];
				
			if(gid1 == 0 || gid1 == m2-1)
				yy += x[(3*m2 + gid1)*mdf + global_id];
			else
				yy += 2.0*x[(3*m2 + gid1)*mdf + global_id];
			
			if(gid1 < m2-1)
				yy -= x[(3*m2 + gid1 + 1)*mdf + global_id];
		}
	}
	
	if(global_id < m1)
		y[(gid2*m2 + gid1)*mdf + global_id] = r3*0.25*yy + r2*h*h*0.5*x[(gid2*m2 + gid1)*mdf + global_id];
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

__kernel void formB(__global _var *b, __global _var *p2, __global _var *l2, __global _var *psi, __global _var *l3, __local _var *lwork, _var r2, _var r3, _var h, int m1, int m2, int mdf, int ndf) {
	const int local_id = get_local_id(0);
	const int global_id = get_global_id(0);
	const int local_size = get_local_size(0);
	const int gid0 = get_group_id(0);
	const int gid1 = get_group_id(1); // y-diretion
	const int gid2 = get_group_id(2); 
	
	const int n1 = m1+1;
	const int n2 = m2+1;
	
	if(m1 <= gid0*local_size)
		return;
	
	_var bb1 = 0.0;
	if(global_id < m1)
		bb1 = r2*p2[(gid2*m2+gid1)*mdf+global_id] + l2[(gid2*m2+gid1)*mdf+global_id];
		
	_var bb2 = 0.0;
		
	if(gid2 == 0) {
		if(gid1+1 < n2-1) {
			_var psi_a = 0.0, l3_a = 0.0;
			if(0 < global_id && global_id < n1-1) {
				psi_a = psi[(gid1+1)*ndf+global_id];
				l3_a = l3[(gid1+1)*ndf+global_id];
				lwork[local_id] = psi_a;
				lwork[local_size+local_id] = l3_a;
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			
			_var psi_b = 0.0, l3_b = 0.0;
			if(global_id+1 < n1-1) {
				if(local_id < local_size-1) {
					psi_b = lwork[local_id+1];
					l3_b = lwork[local_size+local_id+1];
				} else {
					psi_b = psi[(gid1+1)*ndf+global_id+1];
					l3_b = l3[(gid1+1)*ndf+global_id+1];
				}
			}
			
			if(global_id < m1) {
				bb2 += r3*psi_a - l3_a;
				bb2 -= r3*psi_b - l3_b;
			}
		}
	}
	
	if(gid2 == 1) {
		if(0 < gid1) {
			_var psi_a = 0.0, l3_a = 0.0;
			if(0 < global_id && global_id < n1-1) {
				psi_a = psi[gid1*ndf+global_id];
				l3_a = l3[gid1*ndf+global_id];
				lwork[local_id] = psi_a;
				lwork[local_size+local_id] = l3_a;
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			
			_var psi_b = 0.0, l3_b = 0.0;
			if(global_id+1 < n1-1) {
				if(local_id < local_size-1) {
					psi_b = lwork[local_id+1];
					l3_b = lwork[local_size+local_id+1];
				} else {
					psi_b = psi[gid1*ndf+global_id+1];
					l3_b = l3[gid1*ndf+global_id+1];
				}
			}
			
			if(global_id < m1) {
				bb2 += r3*psi_a - l3_a;
				bb2 -= r3*psi_b - l3_b;
			}
		}
	}
		
	if(gid2 == 2) {
		if(0 < global_id && global_id < n1-1) {
			_var psi_a = 0.0, l3_a = 0.0;
			if(0 < gid1) {
				psi_a = psi[gid1*ndf+global_id];
				l3_a = l3[gid1*ndf+global_id];
			}
			
			_var psi_b = 0.0, l3_b = 0.0;
			if(gid1+1 < n2-1) {
				psi_b = psi[(gid1+1)*ndf+global_id];
				l3_b = l3[(gid1+1)*ndf+global_id];
			}
			
			if(global_id < m1) {
				bb2 += r3*psi_a - l3_a;
				bb2 -= r3*psi_b - l3_b;
			}
		}
	}
	
	if(gid2 == 3) {
		if(global_id+1 < n1-1) {
			_var psi_a = 0.0, l3_a = 0.0;
			if(0 < gid1) {
				psi_a = psi[gid1*ndf+global_id+1];
				l3_a = l3[gid1*ndf+global_id+1];
			}
			
			_var psi_b = 0.0, l3_b = 0.0;
			if(gid1+1 < n2-1) {
				psi_b = psi[(gid1+1)*ndf+global_id+1];
				l3_b = l3[(gid1+1)*ndf+global_id+1];
			}
			
			if(global_id < m1) {
				bb2 += r3*psi_a - l3_a;
				bb2 -= r3*psi_b - l3_b;
			}
		}
	}
	
	if(global_id < m1)
		b[(gid2*m2+gid1)*mdf+global_id] = 0.5*h*h*bb1 + 0.5*h*bb2;
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

__kernel void formG(__global _var *g, __global _var *f, __global _var *p1, __global _var *l1, __local _var *lwork, _var r1, _var h, int n1, int n2, int ndf, int mdf, int ndf2) {
	const int local_id = get_local_id(0);
	const int global_id = get_global_id(0);
	const int local_size = get_local_size(0);
	const int gid0 = get_group_id(0);
	const int gid1 = get_group_id(1); // y-diretion
	
	const int m1 = n1-1;
	const int m2 = n2-1;

	// |  /|  /|
	// |/>>|/  |
	// +---+---+
	// |  /|  /|
	// |/  |/  |
	_var2 e1;
	if(global_id < m1) {
		e1.x = r1 * p1[(m2+gid1+1)*mdf+global_id] - l1[(m2+gid1+1)*mdf+global_id];
		lwork[local_id] = e1.x;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	_var e2, e3;
	if(global_id+1 < m1) {
		// |  /|  /|
		// |/^^|/  |
		// +^^^+---+
		// |  /|  /|
		// |/  |/  |
		e1.y = r1 * p1[(3*m2+gid1+1)*mdf+global_id] - l1[(3*m2+gid1+1)*mdf+global_id];
	
	
		// |  /|^^/|
		// |/  |/  |
		// +---+---+
		// |  /|  /|
		// |/  |/  |
		e2 = r1 * p1[(2*m2+gid1+1)*mdf+global_id+1] - l1[(2*m2+gid1+1)*mdf+global_id+1];
	
		// |  /|  /|
		// |/  |/>>|
		// +---+---+
		// |  /|  /|
		// |/  |/  |
		if(local_id < local_size-1)
			e3 = lwork[local_id+1];
		else
			e3 = r1 * p1[(m2+gid1+1)*mdf+global_id+1] - l1[(m2+gid1+1)*mdf+global_id+1];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// |  /|  /|
	// |/  |/  |
	// +---+---+
	// |>>/|  /|
	// |/  |/  |
	_var e4;
	if(global_id < m1) {
		e4 = r1 * p1[gid1*mdf+global_id] - l1[gid1*mdf+global_id];
		lwork[local_id] = e4;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	_var e5; _var2 e6;
	if(global_id+1 < m1) {
	
		// |  /|  /|
		// |/  |/  |
		// +---+---+
		// |  /|  /|
		// |/^^|/  |
		e5 = r1 * p1[(3*m2+gid1)*mdf+global_id] - l1[(3*m2+gid1)*mdf+global_id];
		
		// |  /|  /|
		// |/  |/  |
		// +---+---+
		// |  /|>>/|
		// |/  |/  |
		if(local_id < local_size-1)
			e6.x = lwork[local_id+1];
		else
			e6.x = r1 * p1[gid1*mdf+global_id+1] - l1[gid1*mdf+global_id+1];
			
		// |  /|  /|
		// |/  |/  |
		// +---+---+
		// |  /|^^/|
		// |/  |/  |
		e6.y = r1 * p1[(2*m2+gid1)*mdf+global_id+1] - l1[(2*m2+gid1)*mdf+global_id+1];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	_var tmp = (e1.x - e1.y - e2 - e3 + e4 + e5 - e6.x + e6.y);
	
	_var b = 0;
	if(gid1 == 0)         b += f[global_id+1];
	if(gid1 == n2-3)      b += f[(n2-1)*ndf+global_id+1];

	if(global_id == 0)    b += f[(gid1+1)*ndf];
	if(global_id == n1-3) b += f[(gid1+1)*ndf+n1-1];
	
	if(global_id < n1-2) 
		g[gid1*ndf2+global_id] = b + ( 0.5*h*tmp + h*h*f[(gid1+1)*ndf+global_id+1] ) * INV(r1);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

__kernel void copyU(__global _var *u, __global _var *f, __global _var *g, int n1, int n2, int ndf, int ndf2) {
	const int global_id = get_global_id(0);
	const int global_size = get_global_size(0);
	const int gid1 = get_group_id(1);
	
	if(gid1 == 0 || gid1 == n2-1) {
		for(int i = global_id; i < n1; i += global_size)
			u[gid1*ndf + i] = f[gid1*ndf + i];
		return;
	}

	for(int i = global_id; i < n1; i += global_size) {
		_var value;
		if(i == 0 || i == n1-1)
			value = f[gid1*ndf + i];
		else
			value = g[(gid1-1)*ndf2 + i-1];
			 
		u[gid1*ndf + i] = value;
	}
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

_var d_d0_g(_var x, _var b11, _var b12, _var b21, _var b22, _var r1, _var r2) {
	return (pow2(x)*0.5)*(r1+r2*INV(1+pow2(x)))-x*sqrt(pow2(b11+b21*INV(sqrt(1+pow2(x))))+pow2(b12+b22*INV(sqrt(1+pow2(x)))));
}

_var d_d0_f(_var x, _var y, _var b11, _var b12, _var b21, _var b22, _var r1, _var r2) {
	return ((pow2(x)+pow2(y))*0.5)*(r1+r2*INV(1+pow2(x)+pow2(y)))-((b11+b21*INV(sqrt(1+pow2(x)+pow2(y))))*x+(b12+b22*INV(sqrt(1+pow2(x)+pow2(y))))*y);
}

_var d_dx_f(_var x, _var y, _var b11, _var b12, _var b21, _var b22, _var r1, _var r2) {
	return x*(r1+r2*INV(pow2(x)+pow2(y)+1))-(r2*x*(pow2(x)+pow2(y)))*INV(pow2(pow2(x)+pow2(y)+1))-b11+(b21*pow2(x))*INV(pow3_2(pow2(x)+pow2(y)+1))-b21*INV(sqrt(pow2(x)+pow2(y)+1))+(b22*x*y)*INV(pow3_2(pow2(x)+pow2(y)+1));
}

_var d_dy_f(_var x, _var y, _var b11, _var b12, _var b21, _var b22, _var r1, _var r2) {
	return y*(r1+r2*INV(pow2(x)+pow2(y)+1))-(r2*y*(pow2(x)+pow2(y)))*INV(pow2(pow2(x)+pow2(y)+1))-b12+(b21*x*y)*INV(pow3_2(pow2(x)+pow2(y)+1))+(b22*pow2(y))*INV(pow3_2(pow2(x)+pow2(y)+1))-b22*INV(sqrt(pow2(x)+pow2(y)+1));
}

_var d_dxx_f(_var x, _var y, _var b11, _var b12, _var b21, _var b22, _var r1, _var r2) {
	return r1-(4*r2*pow2(x))*INV(pow2(pow2(x)+pow2(y)+1))+(4*r2*pow2(x)*(pow2(x)+pow2(y)))*INV(pow3(pow2(x)+pow2(y)+1))+r2*INV(pow2(x)+pow2(y)+1)-(r2*(pow2(x)+pow2(y)))*INV(pow2(pow2(x)+pow2(y)+1))+(3*b21*x)*INV(pow3_2(pow2(x)+pow2(y)+1))-(3*b21*pow3(x))*INV(pow5_2(pow2(x)+pow2(y)+1))-(3*b22*pow2(x)*y)*INV(pow5_2(pow2(x)+pow2(y)+1))+(b22*y)*INV(pow3_2(pow2(x)+pow2(y)+1));
}

_var d_dyy_f(_var x, _var y, _var b11, _var b12, _var b21, _var b22, _var r1, _var r2) {
	return r1-(4*r2*pow2(y))*INV(pow2(pow2(x)+pow2(y)+1))+(4*r2*pow2(y)*(pow2(x)+pow2(y)))*INV(pow3(pow2(x)+pow2(y)+1))+r2*INV(pow2(x)+pow2(y)+1)-(r2*(pow2(x)+pow2(y)))*INV(pow2(pow2(x)+pow2(y)+1))-(3*b21*x*pow2(y))*INV(pow5_2(pow2(x)+pow2(y)+1))+(b21*x)*INV(pow3_2(pow2(x)+pow2(y)+1))+(3*b22*y)*INV(pow3_2(pow2(x)+pow2(y)+1))-(3*b22*pow3(y))*INV(pow5_2(pow2(x)+pow2(y)+1));
}

_var d_dxy_f(_var x, _var y, _var b11, _var b12, _var b21, _var b22, _var r1, _var r2) {
	return -(4*r2*x*y)*INV(pow2(pow2(x)+pow2(y)+1))+(4*r2*x*y*(pow2(x)+pow2(y)))*INV(pow3(pow2(x)+pow2(y)+1))-(3*b21*pow2(x)*y)*INV(pow5_2(pow2(x)+pow2(y)+1))+(b21*y)*INV(pow3_2(pow2(x)+pow2(y)+1))+(b22*x)*INV(pow3_2(pow2(x)+pow2(y)+1))-(3*b22*x*pow2(y))*INV(pow5_2(pow2(x)+pow2(y)+1));
}

_var lambda1(_var a, _var b, _var c, _var d) {
	return 0.5 * (-sqrt(a*a-2*a*d+4*b*c+d*d)+a+d);
}

_var lambda2(_var a, _var b, _var c, _var d) {
	return 0.5 * (+sqrt(a*a-2*a*d+4*b*c+d*d)+a+d);
}

_var2 newton(_var2 x0, _var2 b1, _var2 b2, _var r1, _var r2, int iterLimit, _var tol, int* _k) {

	_var2 x = x0;
	_var2 old = x0;

	int k;
	for(k = 1; k <= iterLimit; k++) {
		_var a = d_dxx_f(x.x, x.y, b1.x, b1.y, b2.x, b2.y, r1, r2);
		_var b = d_dxy_f(x.x, x.y, b1.x, b1.y, b2.x, b2.y, r1, r2);
		_var c = b;
		_var d = d_dyy_f(x.x, x.y, b1.x, b1.y, b2.x, b2.y, r1, r2);

		_var dx1 = d_dx_f(x.x, x.y, b1.x, b1.y, b2.x, b2.y, r1, r2);
		_var dx2 = d_dy_f(x.x, x.y, b1.x, b1.y, b2.x, b2.y, r1, r2);

		if(sqrt(dx1*dx1+dx2*dx2) < tol && norm_var2(x-old)*INV(norm_var2(x)) < tol && 0 < lambda1(a,b,c,d) && 0 < lambda2(a,b,c,d))
			break;

		old = x;

		_var det = a*d-b*c;
		x.x -= INV(det)*(d*dx1 - b*dx2);
		x.y -= INV(det)*(a*dx2 - c*dx1);
	}
	*_k = k;

	return x;
}

_var2 bisection(_var _a_x, _var _b_x, _var2 b1, _var2 b2, _var r1, _var r2, _var tol, bool jump) {

	_var a_x = _a_x;
	_var a = d_d0_g(a_x, b1.x, b1.y, b2.x, b2.y, r1, r2);

	_var b_x = _b_x;
	_var b = d_d0_g(b_x, b1.x, b1.y, b2.x, b2.y, r1, r2);

	if(jump) {
		// Infinite looping is extremely rare but possible
		for(int i = 0; i < 10 && b < a; i++) {
			b_x *= 2;
			b = d_d0_g(b_x, b1.x, b1.y, b2.x, b2.y, r1, r2);
		}
	}

	while(tol*fabs(b_x + a_x)*0.5 < b_x - a_x) {
		_var c1_x = a_x + (b_x - a_x)*0.5 - min(1.0E-2, (b_x - a_x)*0.1);
		_var c1 = d_d0_g(c1_x, b1.x, b1.y, b2.x, b2.y, r1, r2);

		_var c2_x = a_x + (b_x - a_x)*0.5 + min(1.0E-2, (b_x - a_x)*0.1);
		_var c2 = d_d0_g(c2_x, b1.x, b1.y, b2.x, b2.y, r1, r2);

		if(c1 <= c2)
			b_x = c2_x;
		else
			a_x = c1_x;
	}

	_var rho = (b_x + a_x)*0.5;

	_var lambda = rho * INV(norm_var2(b1+b2*INV(sqrt(1.0+pow2(rho)))));

	return lambda*(b1+b2*INV(sqrt(1+pow2(rho))));
}

_var checkDirection(_var2 x, _var2 b1, _var2 b2) {
	_var rho = norm_var2(x);
	_var lambda = rho * INV(norm_var2(b1+b2*INV(sqrt(1.0+pow2(rho)))));
	_var2 x0 = lambda*(b1+b2*INV(sqrt(1+pow2(rho))));
	return norm_var2(x-x0)*INV(norm_var2(x0));
}

#define SOLVE_SUB1_ITER_LIMIT 10

__kernel void solve_sub1(
	__global _var *p1,
	__global _var *p2,
	__global _var *u,
	__global _var *p3,
	__global _var *l1,
	__global _var *l2,
	__local  _var *lwork,
	         _var r1,
	         _var r2,
	         _var h,
	         _var tol,
	          int first,
	          int m1,
	          int m2,
	          int ndf,
	          int mdf) {
	          
	const int global_id = get_global_id(0);
	const int local_size = get_local_size(0);
	const int gid0 = get_group_id(0);
	const int gid1 = get_group_id(1);
	const int gid2 = get_group_id(2);
	
	if(m1 <= gid0*local_size)
		return;

	_var2 u_grad = grad(u, lwork, global_id, gid1, gid2, m1+1, m2+1, ndf, h);
	
	if(m1 <= global_id)
		return;
	
	int idx, idy;
	if(gid2 == 0) {
		idx =        gid1*mdf + global_id;
		idy = (2*m2+gid1)*mdf + global_id;
	} else {
		idx =   (m2+gid1)*mdf + global_id;
		idy = (3*m2+gid1)*mdf + global_id;
	}

	// Old p1
	_var2 op1 = (_var2)(p1[idx], p1[idy]);
	
	// New p1 and p2
	_var2 pp1 = 0.0;
	_var2 pp2 = 0.0;
	
	_var2 b1 = r1 * u_grad + (_var2)(l1[idx], l1[idy]);
	_var2 b2 = r2 * (_var2)(p3[idx], p3[idy]) - (_var2)(l2[idx], l2[idy]);

	if(b1.x != 0.0 || b1.y != 0.0 || b2.x != 0.0 || b2.y != 0.0) {

		_var min1Val = INF;
		_var min2Val = INF;
		_var prevVal = d_d0_f(op1.x, op1.y, b1.x, b1.y, b2.x, b2.y, r1, r2);

		_var2 min1, min2;

		int k;
		min1 = newton(op1, b1, b2, r1, r2, SOLVE_SUB1_ITER_LIMIT, tol, &k);
		min1Val = d_d0_f(min1.x, min1.y, b1.x, b1.y, b2.x, b2.y, r1, r2);

		//If Newton failed
		if(first || SOLVE_SUB1_ITER_LIMIT < k || tol < checkDirection(min1, b1, b2) || prevVal < min1Val) {
			_var a = 0.0;
			_var b = 1;
			_var bb = sqrt(2.0)*INV(h);
			_var2 min = 0.0;
			_var minVal = d_d0_f(min.x, min.y, b1.x, b1.y, b2.x, b2.y, r1, r2);
			do {
				_var2 tmp = bisection(a, b, b1, b2, r1, r2, 1.0E-3, false);
				_var tmpVal = d_d0_f(tmp.x, tmp.y, b1.x, b1.y, b2.x, b2.y, r1, r2);
				if(tmpVal < minVal) {
					min = tmp;
					minVal = tmpVal;
				}
				a = b;
				b *= 4;
			} while(b <= bb);
			_var2 tmp = bisection(a, b, b1, b2, r1, r2, 1.0E-3, true);
			_var tmpVal = d_d0_f(tmp.x, tmp.y, b1.x, b1.y, b2.x, b2.y, r1, r2);
			if(tmpVal < minVal)
				min = tmp;
			min2 = bisection(norm_var2(min)*(1.0-1.0E-3), norm_var2(min)*(1.0+1.0E-3), b1, b2, r1, r2, tol, false);
			min2Val = d_d0_f(min2.x, min2.y, b1.x, b1.y, b2.x, b2.y, r1, r2);
		}

		_var2 min;
		_var minVal;
		if(min1Val < min2Val) {
			min = min1;
			minVal = min1Val;
		} else {
			min = min2;
			minVal = min2Val;
		}

		if(minVal <= prevVal)
			pp1 = min;
		else
			pp1 = op1;
		
		pp2 = pp1*INV(sqrt(1.0+dot_s_var2(pp1)));
	}
	
	p1[idx] = pp1.x;
	p1[idy] = pp1.y;
	
	p2[idx] = pp2.x;
	p2[idy] = pp2.y;
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

__kernel void solve_sub3(__global _var *psi, __global _var *p3, __global _var *l3, __local _var *lwork, _var r3, _var eps, _var h, int n1, int n2, int ndf, int mdf) {
	const int global_id = get_global_id(0);
	const int global_size = get_global_size(0);
	const int gid1 = get_group_id(1);
	
	if((gid1 == 0 || gid1 == n2-1) && global_id < n1) {
		psi[gid1*ndf+global_id] = 0.0;
		return;
	}
		
	_var p3_div = div(p3, lwork, global_id, gid1, n1-1, n2-1, mdf, h);
	
	if(global_id < n1) {
		if(global_id == 0 || global_id == n1-1) {
			psi[gid1*ndf+global_id] = 0.0;
			return;
		}
		
		_var x = r3 * p3_div + l3[gid1*ndf+global_id];
		psi[gid1*ndf+global_id] = INV(r3)*sgn(x)*max(0.0, fabs(x)-eps);
	}
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

__kernel void update_l12(__global _var *l1, __global _var *l2, __global _var *u, __global _var *p1, __global _var *p2, __global _var *p3, __local _var *lwork, _var r1, _var r2, _var h, int m1, int m2, int ndf, int mdf) {
	const int global_id = get_global_id(0);
	const int local_size = get_local_size(0);
	const int gid0 = get_group_id(0);
	const int gid1 = get_group_id(1);
	const int gid2 = get_group_id(2);
	
	if(m1 <= gid0*local_size)
		return;

	_var2 u_grad = grad(u, lwork, global_id, gid1, gid2, m1+1, m2+1, ndf, h);
	
	if(m1 <= global_id)
		return;
	
	int idx, idy;
	if(gid2 == 0) {
		idx =        gid1*mdf + global_id;
		idy = (2*m2+gid1)*mdf + global_id;
	} else {
		idx =   (m2+gid1)*mdf + global_id;
		idy = (3*m2+gid1)*mdf + global_id;
	}
	
	l1[idx] += r1 * (u_grad.x - p1[idx]);
	l1[idy] += r1 * (u_grad.y - p1[idy]);
	
	l2[idx] += r2 * (p2[idx] - p3[idx]);
	l2[idy] += r2 * (p2[idy] - p3[idy]);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

__kernel void update_l3(__global _var *l3, __global _var *p3, __global _var *psi, __local _var *lwork, double r3, _var h, int n1, int n2, int ndf, int mdf) {
	const int global_id = get_global_id(0);
	const int global_size = get_global_size(0);
	const int gid1 = get_group_id(1);
		
	_var p3_div = div(p3, lwork, global_id, gid1, n1-1, n2-1, mdf, h);
	if(global_id < n1)
		l3[gid1*ndf+global_id] += r3 * (p3_div - psi[gid1*ndf+global_id]);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

__kernel void compute_obj(__global _var *value, __global _var *u, __global _var *f, __global _var *p1, __global _var *p2, __global _var *p3, __global _var *psi, __global _var *l1, __global _var *l2, __global _var *l3, __local _var *lwork, _var eps, _var r1, _var r2, _var r3, _var h, int n1, int n2, int ndf, int mdf) {
	const int local_id = get_local_id(0);
	const int global_id = get_global_id(0);
	const int gid1 = get_group_id(1);
	
	const int m1 = n1-1;
	const int m2 = n2-1;
	
	_var val1 = 0.0, val2 = 0.0;
	
	_var2 u_grad_odd = grad(u, lwork, global_id, gid1, 0, n1, n2, ndf, h);
	barrier(CLK_LOCAL_MEM_FENCE);
	_var2 u_grad_even = grad(u, lwork, global_id, gid1, 1, n1, n2, ndf, h);
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(global_id < m1) {
		int idx_odd =         gid1*mdf + global_id;
		int idy_odd =  (2*m2+gid1)*mdf + global_id;

		_var2 p1_odd = (_var2)(p1[idx_odd], p1[idy_odd]);
		_var2 p2_odd = (_var2)(p2[idx_odd], p2[idy_odd]);
		_var2 p3_odd = (_var2)(p3[idx_odd], p3[idy_odd]);
		
		_var2 l1_odd = (_var2)(l1[idx_odd], l1[idy_odd]);
		_var2 l2_odd = (_var2)(l2[idx_odd], l2[idy_odd]);
	
		val1 += 0.5 * r1 * dot_s_var2(u_grad_odd - p1_odd);
		val1 += dot_var2(l1_odd, u_grad_odd - p1_odd);
		val1 += 0.5 * r2 * dot_s_var2(p2_odd - p3_odd);
		val1 += dot_var2(l2_odd, p2_odd - p3_odd);
		
		/////////////////////////////////////////////////////////////
		
		int idx_even =   (m2+gid1)*mdf + global_id;
		int idy_even = (3*m2+gid1)*mdf + global_id;
		
		_var2 p1_even = (_var2)(p1[idx_even], p1[idy_even]);
		_var2 p2_even = (_var2)(p2[idx_even], p2[idy_even]);
		_var2 p3_even = (_var2)(p3[idx_even], p3[idy_even]);
		
		_var2 l1_even = (_var2)(l1[idx_even], l1[idy_even]);
		_var2 l2_even = (_var2)(l2[idx_even], l2[idy_even]);
		
		val1 += 0.5 * r1 * dot_s_var2(u_grad_even - p1_even);
		val1 += dot_var2(l1_even, u_grad_even - p1_even);
		val1 += 0.5 * r2 * dot_s_var2(p2_even - p3_even);
		val1 += dot_var2(l2_even, p2_even - p3_even);
	}
	
	if(gid1+1 < n2-1) {
		_var p3_div = div(p3, lwork, global_id+1, gid1+1, m1, m2, mdf, h);
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if(global_id+1 < n1-1) {
			int idx = (gid1+1)*ndf+global_id+1;
			
			_var psi_v = psi[idx];
			_var f_v = f[idx];
			_var u_v = u[idx];
			_var l3_v = l3[idx];
			
			val2 += eps * fabs(psi_v);
			val2 += 0.5 * pow2(f_v - u_v);
			val2 += r3 * 0.5 * pow2(p3_div - psi_v);
			val2 += l3_v * (p3_div - psi_v);
		}
	}
	
	lwork[local_id] = (0.5*val1 + val2)*h*h;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	_var sum = lsum(lwork, local_id, get_local_size(0));
	
	if(local_id == 0)
		value[gid1*get_num_groups(0)+get_group_id(0)] = sum;
}
