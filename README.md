# nanovec-go

A high-performance Go port of [nano-vectordb](https://github.com/gusye1234/nano-vectordb). Single static binary, zero external dependencies.

## Benchmark (100,000 vectors × dim=1024)

Tested on Intel i5-11400H (6 cores / 12 threads):

| Operation      | Python (numpy) | Go (nanovec-go) | Speedup     |
|----------------|---------------|-----------------|-------------|
| Upsert 100k    | ~1809 ms      | ~256 ms         | **7.1×**    |
| Query top-10   | ~34.5 ms      | ~13.5 ms        | **2.6×**    |
| Save to disk   | ~8800 ms      | ~452 ms         | **~19×**    |
| Load from disk | ~3600 ms      | ~297 ms         | **~12×**    |

Query uses **parallel SGEMV** — the matrix is split across all CPU cores,
each computes a local top-K, then results are merged. Upsert parallelizes
normalization the same way. Save/Load use a **binary format** (raw float32
memory dump) instead of JSON, making I/O ~195× faster than the original
JSON serialization.

## How query works

nanovec-go ships its own **hand-written SGEMV kernels** in Go assembly —
AVX2+FMA3 on x86_64, NEON on ARM64. No CGO, no OpenBLAS, no C compiler needed.
On CPUs without AVX2 (pre-2013 x86), a pure-Go fallback with 8-wide loop
unrolling is used automatically.

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
- **Custom AVX2+FMA3 assembly** — hand-written SGEMV kernel, zero dependencies
- **NEON assembly** — ARM64 kernel for Apple Silicon / AWS Graviton
- **Pure-Go fallback** — 8-wide unrolled dot product on unsupported architectures
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

No external dependencies. Builds into a single static binary on all platforms:
```
go build ./...
```

## Files

| File              | Purpose                                          |
|-------------------|--------------------------------------------------|
| `go.mod`          | Go module definition                             |
| `vectordb.go`     | NanoVectorDB + MultiTenantNanoVDB implementation |
| `simd_amd64.go/s` | AVX2+FMA3 SGEMV kernel + CPUID detection (x86_64)|
| `simd_arm64.go/s` | NEON SGEMV kernel (ARM64)                        |
| `simd_generic.go` | Pure-Go fallback (other architectures)            |
| `simd_test.go`    | SIMD correctness tests + dot product benchmarks  |
| `uuid.go`         | stdlib UUID (no external deps)                   |
| `vectordb_test.go`| Unit tests + benchmarks                          |
| `REPORT.md`       | Full technical report and vector DB theory        |
