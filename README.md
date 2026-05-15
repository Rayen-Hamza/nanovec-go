# nanovec-go

A high-performance Go port of [nano-vectordb](https://github.com/gusye1234/nano-vectordb), with OpenBLAS acceleration.

## Benchmark (100,000 vectors × dim=1024, single Xeon core)

| Operation      | Python (numpy) | Go (nanovec-go) | Speedup    |
|----------------|---------------|-----------------|------------|
| Upsert 100k    | ~1809 ms      | ~750 ms         | **2.4×**   |
| Query top-10   | ~34.5 ms      | ~32.9 ms        | **1.05×**  |
| Save to disk   | ~8800 ms      | ~600 ms         | **~15×**   |
| Load from disk | ~3600 ms      | ~350 ms         | **~10×**   |

On **multi-core** machines the gap widens further — Go has no GIL and
parallelizes upsert normalization across all cores, while Python's numpy
is single-threaded.

## How query matches Python

Python's `np.dot(matrix, query)` calls **OpenBLAS SGEMV** under the hood —
hand-written AVX2/AVX-512 assembly for float32 matrix-vector multiply.

nanovec-go calls the **same OpenBLAS kernel** via CGO, so both languages
run identical native code. The result: equal single-core query speed, with
Go winning on throughput from eliminated GIL overhead.

## Architecture

```
Query path (no filter):
  normalize(q) → cblas_sgemv(matrix[N×dim], q) → heap top-K → results
  ↑ one BLAS call over the entire contiguous matrix, zero copies

Query path (with filter):
  normalize(q) → gather rows into 512-row chunks → cblas_sgemv per chunk → heap top-K
  ↑ chunk size fits L2/L3 cache (~2MB per chunk)

Upsert path:
  pre-alloc slab[N×dim] → parallel normalize (NumCPU workers) → bulk append
  ↑ zero per-vector allocations in the hot path
```

## Performance techniques

- **OpenBLAS SGEMV** via CGO — AVX2/AVX-512 matrix-vector multiply, same kernel as numpy
- **Slab allocation** — one `[]float32` for all incoming vectors, no per-item `make()`
- **Worker pool** — normalization chunked across `NumCPU` goroutines
- **O(1) upsert lookup** — `id→rowIndex` hash map vs Python's linear scan
- **Cache-friendly filtered SGEMV** — 512-row gather chunks that fit in L2/L3
- **O(n log k) top-K heap** — min-heap instead of full `O(n log n)` sort
- **Fast auto-ID** — atomic counter instead of MD5-hashing every vector
- **RWMutex** — concurrent reads never block each other

## Usage

```go
import nanovdb "github.com/yourusername/nanovec-go"

// Create or load a DB (loads from file if it exists)
vdb, err := nanovdb.NewNanoVectorDB(1024, "my.json")

// Upsert — batch insert/update
report := vdb.Upsert([]nanovdb.Data{
    {
        nanovdb.FieldVector: []float32{ /* 1024-dim embedding */ },
        nanovdb.FieldID:     "optional-custom-id",   // omit for auto-ID
        "any_field":         "any_value",
    },
})
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
- OpenBLAS: `apt install libopenblas-dev` / `brew install openblas`

## Files

| File              | Purpose                                          |
|-------------------|--------------------------------------------------|
| `vectordb.go`     | NanoVectorDB + MultiTenantNanoVDB implementation |
| `blas.go`         | CGO OpenBLAS SGEMV wrapper                       |
| `uuid.go`         | stdlib UUID (no external deps)                   |
| `vectordb_test.go`| Unit tests + benchmarks                          |
