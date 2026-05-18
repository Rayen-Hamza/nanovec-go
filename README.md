<div align="center">

# nanovec-go

**A high-performance, zero-dependency in-memory vector database for Go.**

[![Go Reference](https://pkg.go.dev/badge/github.com/Rayen-Hamza/nanovec-go.svg)](https://pkg.go.dev/github.com/Rayen-Hamza/nanovec-go)
[![Go Report Card](https://goreportcard.com/badge/github.com/Rayen-Hamza/nanovec-go)](https://goreportcard.com/report/github.com/Rayen-Hamza/nanovec-go)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Go Version](https://img.shields.io/badge/go-%3E%3D1.21-00ADD8.svg)](go.mod)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)](#)

</div>

---

## Highlights

| | Feature | Description |
|---|---|---|
| :zap: | **SIMD Accelerated** | Hand-written AVX2+FMA3 (x86_64) and NEON (ARM64) assembly kernels |
| :lock: | **Thread-Safe** | RWMutex — concurrent reads never block each other |
| :floppy_disk: | **Binary I/O** | Raw float32 memory dump, ~195x faster than JSON serialization |
| :busts_in_silhouette: | **Multi-Tenant** | LRU-evicted tenant management with auto-save on eviction |
| :package: | **Zero Dependencies** | Single `go get`, no CGO, no C compiler needed |
| :dart: | **O(n log k) Top-K** | Min-heap instead of full sort for efficient similarity search |

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Benchmarks](#benchmarks)
- [API Reference](#api-reference)
- [Architecture](#architecture)
- [Advanced Usage](#advanced-usage)
- [Performance Techniques](#performance-techniques)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

```bash
go get github.com/Rayen-Hamza/nanovec-go
```

## Quick Start

```go
package main

import (
    "fmt"
    "log"

    nanovdb "github.com/Rayen-Hamza/nanovec-go"
)

func main() {
    // Create or load a database
    db, err := nanovdb.NewNanoVectorDB(1024, "vectors.db")
    if err != nil {
        log.Fatal(err)
    }

    // Insert vectors
    report, err := db.Upsert([]nanovdb.Data{
        {
            nanovdb.FieldVector: embedding,        // []float32, len=1024
            nanovdb.FieldID:     "doc-1",           // optional (auto-generated if omitted)
            "title":             "Hello World",     // arbitrary metadata
        },
    })

    // Query — cosine similarity, top-K
    results := db.Query(queryVec, nanovdb.QueryOption{TopK: 10})
    for _, r := range results {
        fmt.Printf("%s  score=%.4f\n", r[nanovdb.FieldID], r[nanovdb.FieldMetrics])
    }

    // Persist to disk
    db.Save()
}
```

## Benchmarks

**100,000 vectors x dim=1024** — Intel i5-11400H (6C/12T):

| Operation | Python (numpy) | Go (nanovec-go) | Speedup |
|---|---|---|---|
| Upsert 100k | ~1809 ms | ~256 ms | **7.1x** |
| Query top-10 | ~34.5 ms | ~13.5 ms | **2.6x** |
| Save to disk | ~8800 ms | ~452 ms | **~19x** |
| Load from disk | ~3600 ms | ~297 ms | **~12x** |

## API Reference

### Types

| Type | Description |
|---|---|
| `Data` | `map[string]any` — a row with arbitrary fields plus a vector |
| `UpsertReport` | Lists of inserted and updated IDs |
| `QueryOption` | `TopK`, `BetterThanThreshold`, `FilterFunc` |
| `NanoVectorDB` | Thread-safe in-memory vector database |
| `MultiTenantNanoVDB` | Manages multiple databases with LRU eviction |

### Constants

| Constant | Value | Description |
|---|---|---|
| `FieldID` | `"__id__"` | Reserved key for document ID |
| `FieldVector` | `"__vector__"` | Reserved key for the embedding vector |
| `FieldMetrics` | `"__metrics__"` | Reserved key for query result scores |

### NanoVectorDB

```go
func NewNanoVectorDB(embeddingDim int, storageFile string) (*NanoVectorDB, error)

func (db *NanoVectorDB) Upsert(datas []Data) (UpsertReport, error)
func (db *NanoVectorDB) Query(query []float32, opts QueryOption) []map[string]any
func (db *NanoVectorDB) Get(ids []string) []map[string]any
func (db *NanoVectorDB) Delete(ids []string)
func (db *NanoVectorDB) Save() error
func (db *NanoVectorDB) SaveJSON() error
func (db *NanoVectorDB) Len() int
func (db *NanoVectorDB) GetAdditionalData() map[string]any
func (db *NanoVectorDB) StoreAdditionalData(kv map[string]any)
```

### MultiTenantNanoVDB

```go
func NewMultiTenantNanoVDB(embeddingDim, maxCapacity int, storageDir string) (*MultiTenantNanoVDB, error)

func (m *MultiTenantNanoVDB) CreateTenant() (string, error)
func (m *MultiTenantNanoVDB) GetTenant(id string) (*NanoVectorDB, error)
func (m *MultiTenantNanoVDB) ContainsTenant(id string) bool
func (m *MultiTenantNanoVDB) DeleteTenant(id string)
func (m *MultiTenantNanoVDB) Save() error
```

> :book: Full documentation on [pkg.go.dev](https://pkg.go.dev/github.com/Rayen-Hamza/nanovec-go)

## Architecture

```
Query path (parallel, no filter):
  normalize(q) --> split matrix across NumCPU workers
    --> each: SGEMV(chunk, q) --> local top-K
    --> merge partial top-K --> final results

Query path (with filter):
  normalize(q) --> gather matching rows into 512-row chunks
    --> SGEMV per chunk --> merge top-K
  ^ chunk size fits L2/L3 cache (~2 MB per chunk)

Upsert path:
  validate --> pre-alloc slab[N x dim] --> parallel normalize (NumCPU workers)
    --> lock --> insert/update --> bulk append --> unlock

Save path (binary):
  header (20 bytes) --> raw float32 matrix dump --> JSON metadata
  ^ zero per-element encoding, ~195x faster than JSON
```

## Advanced Usage

### Filtered Queries

```go
threshold := float32(0.7)
results := db.Query(queryVec, nanovdb.QueryOption{
    TopK:                5,
    BetterThanThreshold: &threshold,
    FilterFunc: func(d nanovdb.Data) bool {
        return d["category"] == "science"
    },
})
```

### Multi-Tenancy

```go
mt, _ := nanovdb.NewMultiTenantNanoVDB(1024, 100, "./tenants")

// Create and use tenants
tenantID, _ := mt.CreateTenant()
tenant, _ := mt.GetTenant(tenantID)
tenant.Upsert(vectors)

// LRU eviction happens automatically when capacity is exceeded.
// Evicted tenants are saved to disk and reloaded on next access.

mt.Save() // flush all in-memory tenants
```

### Binary vs JSON Persistence

```go
db.Save()     // binary format (recommended) — ~195x faster
db.SaveJSON() // legacy JSON format — compatible with Python nano-vectordb
```

<details>
<summary><strong>:rocket: Performance Techniques</strong></summary>

<br>

- **Parallel SGEMV** — query split across all CPU cores, local top-K per worker, merged
- **Custom AVX2+FMA3 assembly** — hand-written SGEMV kernel, zero dependencies
- **NEON assembly** — ARM64 kernel for Apple Silicon / AWS Graviton
- **Pure-Go fallback** — 8-wide unrolled dot product on unsupported architectures
- **Binary I/O** — raw float32 memory dump, ~195x faster than JSON serialization
- **Slab allocation** — one `[]float32` for all incoming vectors, no per-item `make()`
- **Worker pool** — normalization chunked across `NumCPU` goroutines
- **O(1) upsert lookup** — `id -> rowIndex` hash map vs Python's linear scan
- **Cache-friendly filtered SGEMV** — 512-row gather chunks that fit in L2/L3
- **O(n log k) top-K heap** — min-heap instead of full `O(n log n)` sort
- **Fast auto-ID** — atomic counter instead of MD5-hashing every vector
- **RWMutex** — concurrent reads never block each other

</details>

## Project Structure

```
nanovec-go/
├── doc.go                 # Package-level godoc
├── types.go               # Public types and constants
├── vectordb.go            # NanoVectorDB — constructor, CRUD, query
├── storage.go             # Binary and JSON persistence
├── multitenant.go         # Multi-tenant manager with LRU eviction
├── math.go                # Vector math, normalization, top-K heap
├── vectordb_test.go       # Integration tests and benchmarks
├── example_test.go        # Testable examples (shown in godoc)
│
├── internal/
│   └── simd/              # SIMD-accelerated dot product kernels
│       ├── scores.go          # Pure-Go fallback
│       ├── scores_amd64.go/s  # AVX2+FMA3 (x86_64)
│       └── scores_arm64.go/s  # NEON (ARM64)
│
└── examples/
    ├── basic/main.go          # Upsert + query + save
    └── multitenant/main.go    # Multi-tenant usage
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

```bash
# Run tests
go test ./...

# Run benchmarks
go test -bench=. -benchtime=3s ./...

# Vet
go vet ./...
```

## License

[MIT](LICENSE) &copy; 2025 Rayen Hamza
