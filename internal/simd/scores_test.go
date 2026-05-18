package simd

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func refDot(matrix, query []float32, n, dim int) []float32 {
	scores := make([]float32, n)
	for i := 0; i < n; i++ {
		var sum float64
		for j := 0; j < dim; j++ {
			sum += float64(matrix[i*dim+j]) * float64(query[j])
		}
		scores[i] = float32(sum)
	}
	return scores
}

func TestScoresCorrectness(t *testing.T) {
	dims := []int{1, 3, 7, 8, 15, 16, 31, 32, 33, 63, 64, 127, 128, 384, 512, 768, 1024, 1536}
	ns := []int{0, 1, 2, 5, 100, 512}
	rng := rand.New(rand.NewSource(42))

	for _, dim := range dims {
		for _, n := range ns {
			matrix := make([]float32, n*dim)
			query := make([]float32, dim)
			for i := range matrix {
				matrix[i] = rng.Float32()*2 - 1
			}
			for i := range query {
				query[i] = rng.Float32()*2 - 1
			}

			got := make([]float32, n)
			Scores(matrix, query, n, dim, got)
			want := refDot(matrix, query, n, dim)

			for i := range got {
				diff := float64(got[i] - want[i])
				tol := 2e-3*math.Abs(float64(want[i])) + 1e-5
				if math.Abs(diff) > tol {
					t.Errorf("dim=%d n=%d i=%d: got %v want %v (diff %v)",
						dim, n, i, got[i], want[i], diff)
				}
			}
		}
	}
}

func BenchmarkDotProduct(b *testing.B) {
	for _, dim := range []int{384, 768, 1024, 1536} {
		rng := rand.New(rand.NewSource(99))
		n := 10000
		matrix := make([]float32, n*dim)
		query := make([]float32, dim)
		scores := make([]float32, n)
		for i := range matrix {
			matrix[i] = rng.Float32()
		}
		for i := range query {
			query[i] = rng.Float32()
		}

		b.Run(fmt.Sprintf("dim=%d", dim), func(b *testing.B) {
			b.SetBytes(int64(n * dim * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Scores(matrix, query, n, dim, scores)
			}
		})
	}
}
