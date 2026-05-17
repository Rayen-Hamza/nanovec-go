package nanovectordb

//go:noescape
func dotNEON(matrix *float32, query *float32, n int, dim int, scores *float32)

func blasScores(matrix []float32, query []float32, n, dim int, scores []float32) {
	if n == 0 {
		return
	}
	dotNEON(&matrix[0], &query[0], n, dim, &scores[0])
}
