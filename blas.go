//go:build cgo

package nanovectordb

/*
#cgo linux,amd64 CFLAGS: -I/usr/include/x86_64-linux-gnu
#cgo linux,amd64 LDFLAGS: -L/usr/lib/x86_64-linux-gnu
#cgo linux,arm64 CFLAGS: -I/usr/include/aarch64-linux-gnu
#cgo linux,arm64 LDFLAGS: -L/usr/lib/aarch64-linux-gnu
#cgo darwin CFLAGS: -I/opt/homebrew/opt/openblas/include -I/usr/local/opt/openblas/include
#cgo darwin LDFLAGS: -L/opt/homebrew/opt/openblas/lib -L/usr/local/opt/openblas/lib
#cgo windows CFLAGS: -I${SRCDIR}/openblas/include
#cgo windows LDFLAGS: -L${SRCDIR}/openblas/lib
#cgo LDFLAGS: -lopenblas -lm
#include "cblas.h"

static void blasSgemv(
    int n, int dim,
    float *matrix,
    float *query,
    float *scores)
{
    cblas_sgemv(
        CblasRowMajor,
        CblasNoTrans,
        n,
        dim,
        1.0f,
        matrix, dim,
        query,  1,
        0.0f,
        scores, 1
    );
}
*/
import "C"
import "unsafe"

// blasScores computes scores[i] = dot(matrix[i*dim:(i+1)*dim], query)
// for all i in [0, n) using OpenBLAS SGEMV in a single call.
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
