package nanovectordb

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"runtime"
	"sync"
	"unsafe"
)

type diskStorage struct {
	EmbeddingDim   int              `json:"embedding_dim"`
	Data           []map[string]any `json:"data"`
	Matrix         []float32        `json:"matrix"`
	AdditionalData map[string]any   `json:"additional_data,omitempty"`
}

var binaryMagic = [4]byte{'N', 'V', 'D', 'B'}

type binaryHeader struct {
	Magic        [4]byte
	Version      uint32
	EmbeddingDim uint32
	NumRows      uint32
	MetaLen      uint32
}

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
		db.additionalData = make(map[string]any)
	}
	for i, row := range db.data {
		if id, ok := row[FieldID].(string); ok {
			db.idToIndex[id] = i
		}
	}
	return nil
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
		Data           []map[string]any `json:"data"`
		AdditionalData map[string]any   `json:"additional_data,omitempty"`
	}
	if err := json.Unmarshal(metaJSON, &meta); err != nil {
		return fmt.Errorf("decode metadata: %w", err)
	}

	db.data = meta.Data
	db.additionalData = meta.AdditionalData
	if db.additionalData == nil {
		db.additionalData = make(map[string]any)
	}
	for i, row := range db.data {
		if id, ok := row[FieldID].(string); ok {
			db.idToIndex[id] = i
		}
	}
	return nil
}

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

// Save persists the database to disk in binary format.
func (db *NanoVectorDB) Save() error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	meta := struct {
		Data           []map[string]any `json:"data"`
		AdditionalData map[string]any   `json:"additional_data,omitempty"`
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
