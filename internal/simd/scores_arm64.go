package simd

//go:noescape
func dotNEON(matrix *float32, query *float32, n int, dim int, scores *float32)

// Scores computes dot(matrix[i], query) for each row i and writes results to scores.
func Scores(matrix []float32, query []float32, n, dim int, scores []float32) {
	if n == 0 {
		return
	}
	dotNEON(&matrix[0], &query[0], n, dim, &scores[0])
}
