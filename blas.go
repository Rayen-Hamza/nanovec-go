package nanovectordb

/*
#cgo CFLAGS:  -I/usr/include/x86_64-linux-gnu
#cgo LDFLAGS: -L/usr/lib/x86_64-linux-gnu -lopenblas -lpthread -lm
#include "cblas.h"

// blasSgemv wraps cblas_sgemv:
//   scores[i] = dot(matrix[i*dim : (i+1)*dim], query)   for i in [0, n)
//
// This is a single BLAS Level-2 call (SGEMV) over the entire matrix at once,
// which lets OpenBLAS use its hand-written AVX2/AVX-512 kernel and internal
// blocking/prefetch strategy — far more efficient than n individual dot calls.
static void blasSgemv(
    int n, int dim,
    float *matrix,   // row-major [n × dim]
    float *query,    // [dim]
    float *scores)   // output [n]
{
    cblas_sgemv(
        CblasRowMajor,   // matrix is row-major
        CblasNoTrans,    // no transpose
        n,               // rows
        dim,             // cols
        1.0f,            // alpha
        matrix, dim,     // A, lda
        query,  1,       // x, incx
        0.0f,            // beta
        scores, 1        // y, incy
    );
}
*/
import "C"
import "unsafe"

// blasScores computes scores[i] = dot(matrix[i*dim:(i+1)*dim], query)
// for all i in [0, n) using OpenBLAS SGEMV in a single call.
// Both matrix and query must already be L2-normalized.
func blasScores(matrix []float32, query []float32, n, dim int, scores []float32) {
	if n == 0 {
		return
	}
	C.blasSgemv(
		C.int(n),
		C.int(dim),
		(*C.float)(unsafe.Pointer(&matrix[0])),
		(*C.float)(unsafe.Pointer(&query[0])),
		(*C.float)(unsafe.Pointer(&scores[0])),
	)
}
