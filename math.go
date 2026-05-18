package nanovectordb

import (
	"container/heap"
	"fmt"
	"math"
	"sort"
	"sync/atomic"

	"github.com/Rayen-Hamza/nanovec-go/internal/simd"
)

func normL2(v []float32) float32 {
	var sum float32
	n, i := len(v), 0
	for ; i <= n-8; i += 8 {
		sum += v[i]*v[i] + v[i+1]*v[i+1] + v[i+2]*v[i+2] + v[i+3]*v[i+3] +
			v[i+4]*v[i+4] + v[i+5]*v[i+5] + v[i+6]*v[i+6] + v[i+7]*v[i+7]
	}
	for ; i < n; i++ {
		sum += v[i] * v[i]
	}
	return float32(math.Sqrt(float64(sum)))
}

func normalizeInPlace(v []float32) {
	n := normL2(v)
	if n == 0 {
		return
	}
	inv := float32(1.0) / n
	for i := range v {
		v[i] *= inv
	}
}

var autoIDSeq uint64

func nextAutoID() string {
	n := atomic.AddUint64(&autoIDSeq, 1)
	return fmt.Sprintf("auto-%016x", n)
}

// ── Min-heap for O(n log k) top-K ──

type scoredIdx struct {
	score float32
	index int
}
type minHeap []scoredIdx

func (h minHeap) Len() int            { return len(h) }
func (h minHeap) Less(i, j int) bool  { return h[i].score < h[j].score }
func (h minHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)         { *h = append(*h, x.(scoredIdx)) }
func (h *minHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

func topK(scores []float32, filterIndex []int, k int) []scoredIdx {
	h := make(minHeap, 0, k+1)
	for i, s := range scores {
		if len(h) < k {
			heap.Push(&h, scoredIdx{s, filterIndex[i]})
		} else if s > h[0].score {
			heap.Pop(&h)
			heap.Push(&h, scoredIdx{s, filterIndex[i]})
		}
	}
	sort.Slice(h, func(a, b int) bool { return h[a].score > h[b].score })
	return h
}

func scoreSgemvFiltered(matrix, query []float32, dim int, filterIndex []int, scores []float32) {
	const chunkRows = 512
	scratch := make([]float32, chunkRows*dim)
	m := len(filterIndex)
	for base := 0; base < m; base += chunkRows {
		end := base + chunkRows
		if end > m {
			end = m
		}
		c := end - base
		for i, row := range filterIndex[base:end] {
			copy(scratch[i*dim:(i+1)*dim], matrix[row*dim:(row+1)*dim])
		}
		simd.Scores(scratch[:c*dim], query, c, dim, scores[base:end])
	}
}
