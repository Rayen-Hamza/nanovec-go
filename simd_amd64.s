#include "textflag.h"

// func cpuHasAVX2FMA() bool
TEXT ·cpuHasAVX2FMA(SB), NOSPLIT, $0-1
	// Check OSXSAVE + FMA3: CPUID(EAX=1)
	MOVL	$1, AX
	XORL	CX, CX
	CPUID
	MOVL	CX, DX              // save ECX

	// Check OSXSAVE (bit 27) — must be set before calling XGETBV
	TESTL	$(1<<27), DX
	JZ	no

	// Check FMA3 (bit 12)
	TESTL	$(1<<12), DX
	JZ	no

	// Check OS has enabled AVX state via XGETBV
	XORL	CX, CX
	XGETBV
	ANDL	$6, AX               // bits 1 (SSE state) + 2 (AVX state)
	CMPL	AX, $6
	JNE	no

	// Check AVX2: CPUID(EAX=7, ECX=0)
	MOVL	$7, AX
	XORL	CX, CX
	CPUID
	TESTL	$(1<<5), BX          // AVX2 bit
	JZ	no

	MOVB	$1, ret+0(FP)
	RET

no:
	MOVB	$0, ret+0(FP)
	RET

// func dotAVX2(matrix *float32, query *float32, n int, dim int, scores *float32)
//
// Computes scores[i] = dot(matrix[i*dim:(i+1)*dim], query[0:dim]) for i in [0,n).
// Uses AVX2 + FMA3: 4 YMM accumulators processing 32 floats per inner iteration.
TEXT ·dotAVX2(SB), NOSPLIT, $0-40
	MOVQ	matrix+0(FP), SI     // SI = matrix pointer
	MOVQ	query+8(FP), DX      // DX = query pointer
	MOVQ	n+16(FP), CX         // CX = n (row count)
	MOVQ	dim+24(FP), DI       // DI = dim
	MOVQ	scores+32(FP), R8    // R8 = scores pointer

	TESTQ	CX, CX
	JZ	done

	MOVQ	DI, R9               // R9 = dim
	SHLQ	$2, R9               // R9 = dim * 4 (byte stride per row)
	XORQ	R10, R10             // R10 = row counter (0..n-1)

outer:
	// Zero 4 YMM accumulators
	VXORPS	Y0, Y0, Y0
	VXORPS	Y1, Y1, Y1
	VXORPS	Y2, Y2, Y2
	VXORPS	Y3, Y3, Y3

	XORQ	R11, R11             // R11 = j (element index within row)
	MOVQ	DI, R12              // R12 = dim
	SUBQ	$31, R12             // R12 = dim - 31 (loop bound for groups of 32)

inner32:
	CMPQ	R11, R12
	JG	tail8

	// Load 32 floats from current matrix row
	VMOVUPS	0(SI)(R11*4), Y4
	VMOVUPS	32(SI)(R11*4), Y5
	VMOVUPS	64(SI)(R11*4), Y6
	VMOVUPS	96(SI)(R11*4), Y7

	// FMA: accumulator += row * query
	VFMADD231PS	0(DX)(R11*4), Y4, Y0
	VFMADD231PS	32(DX)(R11*4), Y5, Y1
	VFMADD231PS	64(DX)(R11*4), Y6, Y2
	VFMADD231PS	96(DX)(R11*4), Y7, Y3

	ADDQ	$32, R11
	JMP	inner32

tail8:
	// Handle remaining elements in groups of 8
	MOVQ	DI, R13
	SUBQ	$7, R13              // R13 = dim - 7

tail8_loop:
	CMPQ	R11, R13
	JG	tail1_init

	VMOVUPS	0(SI)(R11*4), Y4
	VFMADD231PS	0(DX)(R11*4), Y4, Y0

	ADDQ	$8, R11
	JMP	tail8_loop

tail1_init:
	// Accumulate remaining scalar elements into X8 (separate register)
	// to avoid VADDSS zeroing the upper bits of Y0.
	VXORPS	X8, X8, X8

tail1:
	CMPQ	R11, DI
	JGE	reduce

	VMOVSS	(SI)(R11*4), X9
	VMULSS	(DX)(R11*4), X9, X9
	VADDSS	X9, X8, X8

	INCQ	R11
	JMP	tail1

reduce:
	// Sum 4 accumulators: Y0 = Y0 + Y1 + Y2 + Y3
	VADDPS	Y1, Y0, Y0
	VADDPS	Y3, Y2, Y2
	VADDPS	Y2, Y0, Y0

	// Horizontal sum Y0 (8 floats → 1 scalar)
	VEXTRACTF128	$1, Y0, X1   // X1 = high 128 bits
	VADDPS	X1, X0, X0           // X0 = [a+e, b+f, c+g, d+h]
	VHADDPS	X0, X0, X0           // X0 = [a+e+b+f, c+g+d+h, ...]
	VHADDPS	X0, X0, X0           // X0 = [sum, ...]

	// Add tail scalar
	VADDSS	X8, X0, X0

	// Store result
	VMOVSS	X0, (R8)(R10*4)      // scores[row] = sum

	// Next row
	INCQ	R10
	ADDQ	R9, SI               // advance matrix pointer by dim*4 bytes
	CMPQ	R10, CX
	JL	outer

done:
	VZEROUPPER
	RET
