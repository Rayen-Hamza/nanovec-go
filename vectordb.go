package nanovectordb

import (
	"fmt"
	"os"
	"runtime"
	"sync"

	"github.com/Rayen-Hamza/nanovec-go/internal/simd"
)

// NanoVectorDB is a thread-safe, high-performance in-memory vector database
// that persists to disk. Compatible with the Python nano-vectordb storage format.
type NanoVectorDB struct {
	EmbeddingDim int
	StorageFile  string

	mu             sync.RWMutex
	data           []map[string]any
	matrix         []float32      // flat row-major [N × EmbeddingDim]
	additionalData map[string]any
	idToIndex      map[string]int // O(1) id → row lookup
}

// NewNanoVectorDB creates or loads a NanoVectorDB from storageFile.
func NewNanoVectorDB(embeddingDim int, storageFile string) (*NanoVectorDB, error) {
	db := &NanoVectorDB{
		EmbeddingDim:   embeddingDim,
		StorageFile:    storageFile,
		additionalData: make(map[string]any),
		idToIndex:      make(map[string]int),
	}
	if _, err := os.Stat(storageFile); err == nil {
		if err := db.load(); err != nil {
			return nil, err
		}
		db.renormalizeAll()
	} else {
		db.data = []map[string]any{}
		db.matrix = make([]float32, 0)
	}
	return db, nil
}

// Len returns the number of stored vectors.
func (db *NanoVectorDB) Len() int {
	db.mu.RLock()
	defer db.mu.RUnlock()
	return len(db.data)
}

// GetAdditionalData returns the stored extra metadata.
func (db *NanoVectorDB) GetAdditionalData() map[string]any {
	db.mu.RLock()
	defer db.mu.RUnlock()
	return db.additionalData
}

// StoreAdditionalData saves key/value pairs into the DB's metadata.
func (db *NanoVectorDB) StoreAdditionalData(kv map[string]any) {
	db.mu.Lock()
	defer db.mu.Unlock()
	for k, v := range kv {
		db.additionalData[k] = v
	}
}

// Upsert inserts or updates a batch of vectors.
// Normalization is parallelized across CPU cores via a bounded worker pool.
func (db *NanoVectorDB) Upsert(datas []Data) (UpsertReport, error) {
	report := UpsertReport{Update: []string{}, Insert: []string{}}
	if len(datas) == 0 {
		return report, nil
	}
	dim := db.EmbeddingDim

	for i, d := range datas {
		vec, ok := d[FieldVector].([]float32)
		if !ok {
			return report, fmt.Errorf("data[%d]: %s must be []float32", i, FieldVector)
		}
		if len(vec) != dim {
			return report, fmt.Errorf("data[%d]: vector length %d != embedding dim %d", i, len(vec), dim)
		}
		if id, hasID := d[FieldID]; hasID {
			if _, ok := id.(string); !ok {
				return report, fmt.Errorf("data[%d]: %s must be string", i, FieldID)
			}
		}
	}

	slab := make([]float32, len(datas)*dim)

	type entry struct {
		id   string
		meta map[string]any
	}
	entries := make([]entry, len(datas))

	workers := runtime.NumCPU()
	if workers < 1 {
		workers = 1
	}
	chunk := (len(datas) + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		s, e := w*chunk, (w+1)*chunk
		if e > len(datas) {
			e = len(datas)
		}
		if s >= e {
			break
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for idx := start; idx < end; idx++ {
				d := datas[idx]
				row := slab[idx*dim : (idx+1)*dim]
				copy(row, d[FieldVector].([]float32))
				normalizeInPlace(row)

				var idStr string
				if id, ok := d[FieldID]; ok {
					idStr = id.(string)
				} else {
					idStr = nextAutoID()
				}
				meta := make(map[string]any, len(d))
				for k, v := range d {
					if k != FieldVector {
						meta[k] = v
					}
				}
				meta[FieldID] = idStr
				entries[idx] = entry{id: idStr, meta: meta}
			}
		}(s, e)
	}
	wg.Wait()

	db.mu.Lock()
	defer db.mu.Unlock()

	insertSlabs := make([]float32, 0, len(datas)*dim)
	for idx, e := range entries {
		slabRow := slab[idx*dim : (idx+1)*dim]
		if rowIdx, exists := db.idToIndex[e.id]; exists {
			copy(db.matrix[rowIdx*dim:(rowIdx+1)*dim], slabRow)
			db.data[rowIdx] = e.meta
			report.Update = append(report.Update, e.id)
		} else {
			db.data = append(db.data, e.meta)
			insertSlabs = append(insertSlabs, slabRow...)
			report.Insert = append(report.Insert, e.id)
		}
	}
	baseIdx := len(db.matrix) / dim
	db.matrix = append(db.matrix, insertSlabs...)
	for i, id := range report.Insert {
		db.idToIndex[id] = baseIdx + i
	}
	return report, nil
}

// Query finds the top-K most similar vectors by cosine similarity.
// Dot-product scoring is parallelized across CPU cores.
func (db *NanoVectorDB) Query(query []float32, opts QueryOption) []map[string]any {
	if len(query) != db.EmbeddingDim {
		return nil
	}
	if opts.TopK <= 0 {
		opts.TopK = 10
	}
	q := make([]float32, len(query))
	copy(q, query)
	normalizeInPlace(q)

	db.mu.RLock()
	defer db.mu.RUnlock()

	n := len(db.data)
	if n == 0 {
		return nil
	}
	dim := db.EmbeddingDim

	filterIndex := make([]int, 0, n)
	if opts.FilterFunc == nil {
		for i := 0; i < n; i++ {
			filterIndex = append(filterIndex, i)
		}
	} else {
		for i, row := range db.data {
			if opts.FilterFunc(row) {
				filterIndex = append(filterIndex, i)
			}
		}
	}
	m := len(filterIndex)
	if m == 0 {
		return nil
	}

	k := opts.TopK
	if k > m {
		k = m
	}

	const parallelThreshold = 10_000
	workers := runtime.NumCPU()
	if m < parallelThreshold || workers < 2 {
		workers = 1
	}

	var ranked []scoredIdx
	if workers == 1 {
		scores := make([]float32, m)
		if opts.FilterFunc == nil {
			simd.Scores(db.matrix, q, m, dim, scores)
		} else {
			scoreSgemvFiltered(db.matrix, q, dim, filterIndex, scores)
		}
		ranked = topK(scores, filterIndex, k)
	} else {
		chunk := (m + workers - 1) / workers
		partials := make([][]scoredIdx, workers)
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			s, e := w*chunk, (w+1)*chunk
			if e > m {
				e = m
			}
			if s >= e {
				break
			}
			wg.Add(1)
			go func(wIdx, start, end int) {
				defer wg.Done()
				c := end - start
				localScores := make([]float32, c)
				localFilter := filterIndex[start:end]
				if opts.FilterFunc == nil {
					simd.Scores(db.matrix[localFilter[0]*dim:], q, c, dim, localScores)
				} else {
					scoreSgemvFiltered(db.matrix, q, dim, localFilter, localScores)
				}
				partials[wIdx] = topK(localScores, localFilter, k)
			}(w, s, e)
		}
		wg.Wait()

		total := 0
		for _, p := range partials {
			total += len(p)
		}
		merged := make([]float32, 0, total)
		mergedIdx := make([]int, 0, total)
		for _, p := range partials {
			for _, si := range p {
				merged = append(merged, si.score)
				mergedIdx = append(mergedIdx, si.index)
			}
		}
		ranked = topK(merged, mergedIdx, k)
	}

	results := make([]map[string]any, 0, len(ranked))
	for _, r := range ranked {
		if opts.BetterThanThreshold != nil && r.score < *opts.BetterThanThreshold {
			break
		}
		row := db.data[r.index]
		res := make(map[string]any, len(row)+1)
		for k, v := range row {
			res[k] = v
		}
		res[FieldMetrics] = r.score
		results = append(results, res)
	}
	return results
}

// Get retrieves metadata rows by ID (no vector).
func (db *NanoVectorDB) Get(ids []string) []map[string]any {
	db.mu.RLock()
	defer db.mu.RUnlock()
	results := make([]map[string]any, 0, len(ids))
	for _, id := range ids {
		if idx, ok := db.idToIndex[id]; ok {
			results = append(results, db.data[idx])
		}
	}
	return results
}

// Delete removes vectors by ID, compacting the matrix.
func (db *NanoVectorDB) Delete(ids []string) {
	db.mu.Lock()
	defer db.mu.Unlock()
	toDelete := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		toDelete[id] = struct{}{}
	}
	dim := db.EmbeddingDim
	newData := make([]map[string]any, 0, len(db.data))
	newMatrix := make([]float32, 0, len(db.data)*dim)
	newIndex := make(map[string]int, len(db.data))
	for i, row := range db.data {
		id, _ := row[FieldID].(string)
		if _, del := toDelete[id]; del {
			continue
		}
		newIndex[id] = len(newData)
		newData = append(newData, row)
		newMatrix = append(newMatrix, db.matrix[i*dim:(i+1)*dim]...)
	}
	db.data, db.matrix, db.idToIndex = newData, newMatrix, newIndex
}
