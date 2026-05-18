#include "textflag.h"

// func dotNEON(matrix *float32, query *float32, n int, dim int, scores *float32)
//
// Computes scores[i] = dot(matrix[i*dim:(i+1)*dim], query[0:dim]) for i in [0,n).
// Uses NEON FMLA: 4 vector accumulators processing 16 floats per inner iteration.
TEXT ·dotNEON(SB), NOSPLIT, $0-40
	MOVD	matrix+0(FP), R0     // R0 = matrix pointer
	MOVD	query+8(FP), R1      // R1 = query pointer
	MOVD	n+16(FP), R2         // R2 = n (row count)
	MOVD	dim+24(FP), R3       // R3 = dim
	MOVD	scores+32(FP), R4    // R4 = scores pointer

	CBZ	R2, done

	MOVD	R3, R5               // R5 = dim
	LSL	$2, R5, R5           // R5 = dim * 4 (byte stride per row)
	MOVD	ZR, R6               // R6 = row counter (0..n-1)

outer:
	// Zero 4 NEON accumulators (V0-V3, each 4×float32)
	VEOR	V0.B16, V0.B16, V0.B16
	VEOR	V1.B16, V1.B16, V1.B16
	VEOR	V2.B16, V2.B16, V2.B16
	VEOR	V3.B16, V3.B16, V3.B16

	MOVD	ZR, R7               // R7 = j (element index within row)
	MOVD	R3, R8               // R8 = dim
	SUB	$15, R8, R8          // R8 = dim - 15 (loop bound for groups of 16)

inner16:
	CMP	R8, R7
	BGT	tail4

	// Compute byte offsets
	LSL	$2, R7, R9           // R9 = j * 4

	// Load 16 floats from current matrix row
	ADD	R0, R9, R10          // R10 = &matrix[row][j]
	VLD1.P	16(R10), [V4.S4]
	VLD1.P	16(R10), [V5.S4]
	VLD1.P	16(R10), [V6.S4]
	VLD1	(R10), [V7.S4]

	// Load 16 floats from query
	ADD	R1, R9, R10          // R10 = &query[j]
	VLD1.P	16(R10), [V16.S4]
	VLD1.P	16(R10), [V17.S4]
	VLD1.P	16(R10), [V18.S4]
	VLD1	(R10), [V19.S4]

	// FMA: accumulator += row * query
	FMLA	V16.S4, V4.S4, V0.S4
	FMLA	V17.S4, V5.S4, V1.S4
	FMLA	V18.S4, V6.S4, V2.S4
	FMLA	V19.S4, V7.S4, V3.S4

	ADD	$16, R7, R7
	B	inner16

tail4:
	MOVD	R3, R8
	SUB	$3, R8, R8           // R8 = dim - 3

tail4_loop:
	CMP	R8, R7
	BGT	tail1

	LSL	$2, R7, R9
	ADD	R0, R9, R10
	VLD1	(R10), [V4.S4]
	ADD	R1, R9, R10
	VLD1	(R10), [V16.S4]
	FMLA	V16.S4, V4.S4, V0.S4

	ADD	$4, R7, R7
	B	tail4_loop

tail1:
	// Scalar tail for remaining elements
	VEOR	V8.B16, V8.B16, V8.B16  // scalar accumulator

tail1_loop:
	CMP	R3, R7
	BGE	reduce

	LSL	$2, R7, R9
	ADD	R0, R9, R10
	FMOVS	(R10), F9
	ADD	R1, R9, R10
	FMOVS	(R10), F10
	FMULS	F10, F9, F9
	FADDS	F9, F8, F8

	ADD	$1, R7, R7
	B	tail1_loop

reduce:
	// Sum 4 accumulators: V0 = V0 + V1 + V2 + V3
	FADD	V1.S4, V0.S4, V0.S4
	FADD	V3.S4, V2.S4, V2.S4
	FADD	V2.S4, V0.S4, V0.S4

	// Horizontal sum V0 (4 floats → 1 scalar)
	FADDP	V0.S4, V0.S4, V0.S4  // pairwise add: [a+b, c+d, ...]
	FADDP	V0.S2, V0.S2, V0.S2  // final scalar in S0

	// Add tail scalar
	FADDS	F8, F0, F0

	// Store result
	LSL	$2, R6, R9
	ADD	R4, R9, R10
	FMOVS	F0, (R10)            // scores[row] = sum

	// Next row
	ADD	$1, R6, R6
	ADD	R5, R0, R0           // advance matrix pointer by dim*4 bytes
	CMP	R2, R6
	BLT	outer

done:
	RET
