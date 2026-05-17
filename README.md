# nanovec-go

A high-performance Go port of [nano-vectordb](https://github.com/gusye1234/nano-vectordb), with OpenBLAS acceleration.

## Benchmark (100,000 vectors × dim=1024)

Tested on Intel i5-11400H (6 cores / 12 threads), pure-Go mode (`CGO_ENABLED=0`):

| Operation      | Python (numpy) | Go (nanovec-go) | Speedup     |
|----------------|---------------|-----------------|-------------|
| Upsert 100k    | ~1809 ms      | ~333 ms         | **5.4×**    |
| Query top-10   | ~34.5 ms      | ~14.2 ms        | **2.4×**    |
| Save to disk   | ~8800 ms      | ~452 ms         | **~19×**    |
| Load from disk | ~3600 ms      | ~290 ms         | **~12×**    |

Query uses **parallel SGEMV** — the matrix is split across all CPU cores,
each computes a local top-K, then results are merged. Upsert parallelizes
normalization the same way. Save/Load use a **binary format** (raw float32
memory dump) instead of JSON, making I/O ~195× faster than the original
JSON serialization.

## How query works

With OpenBLAS enabled, nanovec-go calls the **same SGEMV kernel** as numpy
(hand-written AVX2/AVX-512 assembly). Without OpenBLAS, a pure-Go fallback
with 8-wide loop unrolling is used automatically (`CGO_ENABLED=0`).

Queries are parallelized: the matrix is split across `NumCPU` workers, each
runs SGEMV + local top-K on its chunk, then partial results are merged.

## Architecture

```
Query path (no filter, parallel):
  normalize(q) → split matrix across NumCPU workers
    → each: SGEMV(chunk, q) → local top-K
    → merge partial top-K → final results

Query path (with filter):
  normalize(q) → gather matching rows into 512-row chunks
    → SGEMV per chunk → merge top-K
  ↑ chunk size fits L2/L3 cache (~2MB per chunk)

Upsert path:
  validate → pre-alloc slab[N×dim] → parallel normalize (NumCPU workers)
    → lock → insert/update → bulk append → unlock

Save path:
  binary header (20 bytes) → raw float32 matrix dump → JSON metadata
  ↑ zero per-element encoding, ~195× faster than JSON
```

## Performance techniques

- **Parallel SGEMV** — query split across all CPU cores, local top-K per worker, merged
- **OpenBLAS SGEMV** via CGO — AVX2/AVX-512 matrix-vector multiply (optional)
- **Pure-Go fallback** — 8-wide unrolled dot product when CGO is disabled
- **Binary I/O** — raw float32 memory dump, ~195× faster than JSON serialization
- **Slab allocation** — one `[]float32` for all incoming vectors, no per-item `make()`
- **Worker pool** — normalization chunked across `NumCPU` goroutines
- **O(1) upsert lookup** — `id→rowIndex` hash map vs Python's linear scan
- **Cache-friendly filtered SGEMV** — 512-row gather chunks that fit in L2/L3
- **O(n log k) top-K heap** — min-heap instead of full `O(n log n)` sort
- **Fast auto-ID** — atomic counter instead of MD5-hashing every vector
- **RWMutex** — concurrent reads never block each other

## Usage

```go
import nanovdb "github.com/Rayen-Hamza/nanovec-go"

// Create or load a DB (loads from file if it exists)
vdb, err := nanovdb.NewNanoVectorDB(1024, "my.json")

// Upsert — batch insert/update (validates vector types and dimensions)
report, err := vdb.Upsert([]nanovdb.Data{
    {
        nanovdb.FieldVector: []float32{ /* 1024-dim embedding */ },
        nanovdb.FieldID:     "optional-custom-id",   // omit for auto-ID
        "any_field":         "any_value",
    },
})
if err != nil { /* handle validation error */ }
fmt.Println(report.Insert, report.Update)

// Query — cosine similarity top-K
results := vdb.Query(queryVec, nanovdb.QueryOption{
    TopK: 10,
    BetterThanThreshold: func() *float32 { v := float32(0.5); return &v }(),
    FilterFunc: func(d nanovdb.Data) bool {
        return d["category"] == "science"
    },
})
for _, r := range results {
    fmt.Println(r[nanovdb.FieldID], r[nanovdb.FieldMetrics])
}

// Get by IDs
rows := vdb.Get([]string{"id-1", "id-2"})

// Delete
vdb.Delete([]string{"id-1"})

// Persist to disk
if err := vdb.Save(); err != nil { ... }

// Multi-tenancy with LRU eviction
mt, _ := nanovdb.NewMultiTenantNanoVDB(1024, 100, "./tenants")
tenantID, _ := mt.CreateTenant()
tenant, _ := mt.GetTenant(tenantID)
tenant.Upsert(...)
mt.Save()
```

## Requirements

- Go 1.21+
- OpenBLAS (optional — a pure-Go fallback is used when CGO is disabled):
  - **Debian/Ubuntu:** `apt install libopenblas-dev`
  - **Fedora/RHEL:** `dnf install openblas-devel`
  - **macOS (Homebrew):** `brew install openblas`
  - **Without OpenBLAS:** `CGO_ENABLED=0 go build ./...` (uses pure-Go dot product, slower)

## Files

| File              | Purpose                                          |
|-------------------|--------------------------------------------------|
| `go.mod`          | Go module definition                             |
| `vectordb.go`     | NanoVectorDB + MultiTenantNanoVDB implementation |
| `blas.go`         | CGO OpenBLAS SGEMV wrapper (linux, macOS, win)   |
| `blas_nocgo.go`   | Pure-Go fallback when CGO is disabled            |
| `uuid.go`         | stdlib UUID (no external deps)                   |
| `vectordb_test.go`| Unit tests + benchmarks                          |
| `REPORT.md`       | Full technical report and vector DB theory        |
