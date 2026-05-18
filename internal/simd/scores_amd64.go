package simd

var useAVX2FMA bool

func init() {
	useAVX2FMA = cpuHasAVX2FMA()
}

func cpuHasAVX2FMA() bool

//go:noescape
func dotAVX2(matrix *float32, query *float32, n int, dim int, scores *float32)

// Scores computes dot(matrix[i], query) for each row i and writes results to scores.
func Scores(matrix []float32, query []float32, n, dim int, scores []float32) {
	if n == 0 {
		return
	}
	if useAVX2FMA {
		dotAVX2(&matrix[0], &query[0], n, dim, &scores[0])
		return
	}
	for i := 0; i < n; i++ {
		row := matrix[i*dim : (i+1)*dim]
		var sum float32
		j := 0
		for ; j <= dim-8; j += 8 {
			sum += row[j]*query[j] + row[j+1]*query[j+1] + row[j+2]*query[j+2] + row[j+3]*query[j+3] +
				row[j+4]*query[j+4] + row[j+5]*query[j+5] + row[j+6]*query[j+6] + row[j+7]*query[j+7]
		}
		for ; j < dim; j++ {
			sum += row[j] * query[j]
		}
		scores[i] = sum
	}
}
