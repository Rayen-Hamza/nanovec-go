// Package nanovectordb is a high-performance Go port of Python nano-vectordb.
//
// Performance improvements over the Python version:
//   - No GIL: true goroutine parallelism
//   - Chunked worker-pool normalization (no per-item goroutine overhead)
//   - O(n log k) heap-based top-K instead of O(n log n) full sort
//   - 8-wide loop-unrolled dot product (CPU-cache friendly)
//   - id→rowIndex map for O(1) upsert lookup (Python scans linearly)
//   - RWMutex: concurrent reads don't block each other
//   - Fast auto-ID (atomic counter) avoids expensive MD5 when no ID given
package nanovectordb

import (
	"bufio"
	"container/heap"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"unsafe"
)

// ──────────────────────────────────────────────
// Public field name constants
// ──────────────────────────────────────────────

const (
	FieldID      = "__id__"
	FieldVector  = "__vector__"
	FieldMetrics = "__metrics__"
)

// Data is a row: arbitrary string→value fields plus FieldVector ([]float32).
type Data map[string]interface{}

// UpsertReport mirrors the Python return value.
type UpsertReport struct {
	Update []string `json:"update"`
	Insert []string `json:"insert"`
}

// QueryOption configures a Query call.
type QueryOption struct {
	TopK                int
	BetterThanThreshold *float32
	FilterFunc          func(Data) bool
}

// ──────────────────────────────────────────────
// Math helpers
// ──────────────────────────────────────────────

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

// autoIDSeq is a global monotonic counter used for fast default IDs.
// This avoids 400MB of MD5 hashing for 100k vectors.
var autoIDSeq uint64

func nextAutoID() string {
	n := atomic.AddUint64(&autoIDSeq, 1)
	return fmt.Sprintf("auto-%016x", n)
}

// ──────────────────────────────────────────────
// Min-heap for O(n log k) top-K
// ──────────────────────────────────────────────

type scoredIdx struct {
	score float32
	index int
}
type minHeap []scoredIdx

func (h minHeap) Len() int            { return len(h) }
func (h minHeap) Less(i, j int) bool  { return h[i].score < h[j].score }
func (h minHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x interface{}) { *h = append(*h, x.(scoredIdx)) }
func (h *minHeap) Pop() interface{} {
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

// ──────────────────────────────────────────────
// Disk storage
// ──────────────────────────────────────────────

type diskStorage struct {
	EmbeddingDim   int                      `json:"embedding_dim"`
	Data           []map[string]interface{} `json:"data"`
	Matrix         []float32                `json:"matrix"`
	AdditionalData map[string]interface{}   `json:"additional_data,omitempty"`
}

// ──────────────────────────────────────────────
// NanoVectorDB
// ──────────────────────────────────────────────

// NanoVectorDB is a thread-safe, high-performance in-memory vector database
// that persists to a JSON file.  Compatible with the Python nano-vectordb storage format.
type NanoVectorDB struct {
	EmbeddingDim int
	StorageFile  string

	mu             sync.RWMutex
	data           []map[string]interface{}
	matrix         []float32      // flat row-major [N × EmbeddingDim]
	additionalData map[string]interface{}
	idToIndex      map[string]int // O(1) id → row lookup
}

// NewNanoVectorDB creates or loads a NanoVectorDB from storageFile.
func NewNanoVectorDB(embeddingDim int, storageFile string) (*NanoVectorDB, error) {
	db := &NanoVectorDB{
		EmbeddingDim:   embeddingDim,
		StorageFile:    storageFile,
		additionalData: make(map[string]interface{}),
		idToIndex:      make(map[string]int),
	}
	if _, err := os.Stat(storageFile); err == nil {
		if err := db.load(); err != nil {
			return nil, err
		}
		db.renormalizeAll()
	} else {
		db.data = []map[string]interface{}{}
		db.matrix = make([]float32, 0)
	}
	return db, nil
}

var binaryMagic = [4]byte{'N', 'V', 'D', 'B'}

func (db *NanoVectorDB) load() error {
	f, err := os.Open(db.StorageFile)
	if err != nil {
		return err
	}
	defer f.Close()

	var magic [4]byte
	if _, err := io.ReadFull(f, magic[:]); err != nil {
		return err
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return err
	}

	if magic == binaryMagic {
		return db.loadBinary(f)
	}
	return db.loadJSON(f)
}

func (db *NanoVectorDB) loadJSON(f *os.File) error {
	var ds diskStorage
	if err := json.NewDecoder(f).Decode(&ds); err != nil {
		return err
	}
	if ds.EmbeddingDim != db.EmbeddingDim {
		return fmt.Errorf("embedding dim mismatch: want %d, got %d", db.EmbeddingDim, ds.EmbeddingDim)
	}
	db.data = ds.Data
	db.matrix = ds.Matrix
	db.additionalData = ds.AdditionalData
	if db.additionalData == nil {
		db.additionalData = make(map[string]interface{})
	}
	for i, row := range db.data {
		if id, ok := row[FieldID].(string); ok {
			db.idToIndex[id] = i
		}
	}
	return nil
}

type binaryHeader struct {
	Magic        [4]byte
	Version      uint32
	EmbeddingDim uint32
	NumRows      uint32
	MetaLen      uint32
}

func (db *NanoVectorDB) loadBinary(f *os.File) error {
	r := bufio.NewReaderSize(f, 1<<20)

	var hdr binaryHeader
	if err := binary.Read(r, binary.LittleEndian, &hdr); err != nil {
		return fmt.Errorf("read header: %w", err)
	}
	if hdr.Version != 1 {
		return fmt.Errorf("unsupported binary version %d", hdr.Version)
	}
	if int(hdr.EmbeddingDim) != db.EmbeddingDim {
		return fmt.Errorf("embedding dim mismatch: want %d, got %d", db.EmbeddingDim, hdr.EmbeddingDim)
	}

	nFloats := int(hdr.NumRows) * int(hdr.EmbeddingDim)
	db.matrix = make([]float32, nFloats)
	if nFloats > 0 {
		matrixBytes := unsafe.Slice((*byte)(unsafe.Pointer(&db.matrix[0])), nFloats*4)
		if _, err := io.ReadFull(r, matrixBytes); err != nil {
			return fmt.Errorf("read matrix: %w", err)
		}
	}

	metaJSON := make([]byte, hdr.MetaLen)
	if _, err := io.ReadFull(r, metaJSON); err != nil {
		return fmt.Errorf("read metadata: %w", err)
	}

	var meta struct {
		Data           []map[string]interface{} `json:"data"`
		AdditionalData map[string]interface{}   `json:"additional_data,omitempty"`
	}
	if err := json.Unmarshal(metaJSON, &meta); err != nil {
		return fmt.Errorf("decode metadata: %w", err)
	}

	db.data = meta.Data
	db.additionalData = meta.AdditionalData
	if db.additionalData == nil {
		db.additionalData = make(map[string]interface{})
	}
	for i, row := range db.data {
		if id, ok := row[FieldID].(string); ok {
			db.idToIndex[id] = i
		}
	}
	return nil
}

// renormalizeAll normalizes all vectors using a worker pool.
func (db *NanoVectorDB) renormalizeAll() {
	n := len(db.data)
	if n == 0 {
		return
	}
	dim := db.EmbeddingDim
	workers := runtime.NumCPU()
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		s, e := w*chunk, (w+1)*chunk
		if e > n {
			e = n
		}
		if s >= e {
			break
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				normalizeInPlace(db.matrix[i*dim : (i+1)*dim])
			}
		}(s, e)
	}
	wg.Wait()
}

// Len returns the number of stored vectors.
func (db *NanoVectorDB) Len() int {
	db.mu.RLock()
	defer db.mu.RUnlock()
	return len(db.data)
}

// GetAdditionalData returns the stored extra metadata.
func (db *NanoVectorDB) GetAdditionalData() map[string]interface{} {
	db.mu.RLock()
	defer db.mu.RUnlock()
	return db.additionalData
}

// StoreAdditionalData saves key/value pairs into the DB's metadata.
func (db *NanoVectorDB) StoreAdditionalData(kv map[string]interface{}) {
	db.mu.Lock()
	defer db.mu.Unlock()
	for k, v := range kv {
		db.additionalData[k] = v
	}
}

// Upsert inserts or updates a batch of vectors.
// Normalization is done in a bounded worker pool (no per-item goroutine overhead).
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

	// Pre-allocate a flat slab for all incoming vectors (avoids per-item alloc).
	// Workers normalize directly into the slab.
	slab := make([]float32, len(datas)*dim)

	type entry struct {
		id   string
		meta map[string]interface{}
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
				meta := make(map[string]interface{}, len(d))
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

// Get retrieves metadata rows by ID (no vector).
func (db *NanoVectorDB) Get(ids []string) []map[string]interface{} {
	db.mu.RLock()
	defer db.mu.RUnlock()
	results := make([]map[string]interface{}, 0, len(ids))
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
	newData := make([]map[string]interface{}, 0, len(db.data))
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

// Save persists the database to disk in binary format.
func (db *NanoVectorDB) Save() error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	meta := struct {
		Data           []map[string]interface{} `json:"data"`
		AdditionalData map[string]interface{}   `json:"additional_data,omitempty"`
	}{db.data, db.additionalData}
	metaJSON, err := json.Marshal(meta)
	if err != nil {
		return err
	}

	f, err := os.Create(db.StorageFile)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 1<<20)

	hdr := binaryHeader{
		Magic:        binaryMagic,
		Version:      1,
		EmbeddingDim: uint32(db.EmbeddingDim),
		NumRows:      uint32(len(db.data)),
		MetaLen:      uint32(len(metaJSON)),
	}
	if err := binary.Write(w, binary.LittleEndian, &hdr); err != nil {
		return err
	}

	nFloats := len(db.matrix)
	if nFloats > 0 {
		matrixBytes := unsafe.Slice((*byte)(unsafe.Pointer(&db.matrix[0])), nFloats*4)
		if _, err := w.Write(matrixBytes); err != nil {
			return err
		}
	}

	if _, err := w.Write(metaJSON); err != nil {
		return err
	}
	return w.Flush()
}

// SaveJSON persists the database in the legacy JSON format.
func (db *NanoVectorDB) SaveJSON() error {
	db.mu.RLock()
	defer db.mu.RUnlock()
	ds := diskStorage{
		EmbeddingDim:   db.EmbeddingDim,
		Data:           db.data,
		Matrix:         db.matrix,
		AdditionalData: db.additionalData,
	}
	f, err := os.Create(db.StorageFile)
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewEncoder(f).Encode(ds)
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
		blasScores(scratch[:c*dim], query, c, dim, scores[base:end])
	}
}

// Query finds the top-K most similar vectors (cosine similarity).
// Dot-product scoring is parallelized across CPU cores.
func (db *NanoVectorDB) Query(query []float32, opts QueryOption) []map[string]interface{} {
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

	// Build filter index
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
			blasScores(db.matrix, q, m, dim, scores)
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
					blasScores(db.matrix[localFilter[0]*dim:], q, c, dim, localScores)
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

	results := make([]map[string]interface{}, 0, len(ranked))
	for _, r := range ranked {
		if opts.BetterThanThreshold != nil && r.score < *opts.BetterThanThreshold {
			break
		}
		row := db.data[r.index]
		res := make(map[string]interface{}, len(row)+1)
		for k, v := range row {
			res[k] = v
		}
		res[FieldMetrics] = r.score
		results = append(results, res)
	}
	return results
}

// ──────────────────────────────────────────────
// MultiTenantNanoVDB
// ──────────────────────────────────────────────

// MultiTenantNanoVDB manages multiple NanoVectorDB instances with LRU eviction.
type MultiTenantNanoVDB struct {
	EmbeddingDim int
	MaxCapacity  int
	StorageDir   string

	mu       sync.Mutex
	tenants  map[string]*NanoVectorDB
	lruQueue []string
}

// NewMultiTenantNanoVDB creates a multi-tenant manager.
func NewMultiTenantNanoVDB(embeddingDim, maxCapacity int, storageDir string) (*MultiTenantNanoVDB, error) {
	if maxCapacity < 1 {
		return nil, fmt.Errorf("maxCapacity must be >= 1")
	}
	if err := os.MkdirAll(storageDir, 0o755); err != nil {
		return nil, err
	}
	return &MultiTenantNanoVDB{
		EmbeddingDim: embeddingDim,
		MaxCapacity:  maxCapacity,
		StorageDir:   storageDir,
		tenants:      make(map[string]*NanoVectorDB),
	}, nil
}

func (m *MultiTenantNanoVDB) jsonFile(id string) string {
	return filepath.Join(m.StorageDir, fmt.Sprintf("nanovdb_%s.json", id))
}

// ContainsTenant checks if a tenant exists in memory or on disk.
func (m *MultiTenantNanoVDB) ContainsTenant(id string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.tenants[id]; ok {
		return true
	}
	_, err := os.Stat(m.jsonFile(id))
	return err == nil
}

func (m *MultiTenantNanoVDB) touchLRU(id string) {
	for i, qid := range m.lruQueue {
		if qid == id {
			m.lruQueue = append(m.lruQueue[:i], m.lruQueue[i+1:]...)
			break
		}
	}
	m.lruQueue = append(m.lruQueue, id)
}

func (m *MultiTenantNanoVDB) evict() error {
	if len(m.tenants) < m.MaxCapacity {
		return nil
	}
	oldest := m.lruQueue[0]
	m.lruQueue = m.lruQueue[1:]
	if vdb, ok := m.tenants[oldest]; ok {
		if err := vdb.Save(); err != nil {
			return err
		}
		delete(m.tenants, oldest)
	}
	return nil
}

// CreateTenant creates a new tenant and returns its ID.
func (m *MultiTenantNanoVDB) CreateTenant() (string, error) {
	id := newUUID()
	m.mu.Lock()
	defer m.mu.Unlock()
	if err := m.evict(); err != nil {
		return "", err
	}
	vdb, err := NewNanoVectorDB(m.EmbeddingDim, m.jsonFile(id))
	if err != nil {
		return "", err
	}
	m.tenants[id] = vdb
	m.lruQueue = append(m.lruQueue, id)
	return id, nil
}

// GetTenant returns a tenant's DB, loading from disk if needed.
func (m *MultiTenantNanoVDB) GetTenant(id string) (*NanoVectorDB, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if vdb, ok := m.tenants[id]; ok {
		m.touchLRU(id)
		return vdb, nil
	}
	if _, err := os.Stat(m.jsonFile(id)); err != nil {
		return nil, fmt.Errorf("tenant %s not found", id)
	}
	if err := m.evict(); err != nil {
		return nil, err
	}
	vdb, err := NewNanoVectorDB(m.EmbeddingDim, m.jsonFile(id))
	if err != nil {
		return nil, err
	}
	m.tenants[id] = vdb
	m.lruQueue = append(m.lruQueue, id)
	return vdb, nil
}

// DeleteTenant removes a tenant from memory and disk.
func (m *MultiTenantNanoVDB) DeleteTenant(id string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.tenants, id)
	for i, qid := range m.lruQueue {
		if qid == id {
			m.lruQueue = append(m.lruQueue[:i], m.lruQueue[i+1:]...)
			break
		}
	}
	_ = os.Remove(m.jsonFile(id))
}

// Save flushes all in-memory tenants to disk.
func (m *MultiTenantNanoVDB) Save() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, vdb := range m.tenants {
		if err := vdb.Save(); err != nil {
			return err
		}
	}
	return nil
}
