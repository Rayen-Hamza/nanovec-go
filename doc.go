// Package nanovectordb is a high-performance, zero-dependency in-memory
// vector database for Go.
//
// It provides cosine-similarity search over normalized float32 vectors with
// SIMD-accelerated dot products (AVX2+FMA3 on x86_64, NEON on ARM64), binary
// persistence, and multi-tenant support with LRU eviction.
//
// # Quick Start
//
//	db, err := nanovectordb.NewNanoVectorDB(384, "vectors.db")
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	db.Upsert([]nanovectordb.Data{
//	    {nanovectordb.FieldVector: embedding, "label": "example"},
//	})
//
//	results := db.Query(queryVec, nanovectordb.QueryOption{TopK: 5})
//
// # Features
//
//   - Thread-safe with RWMutex (concurrent reads never block)
//   - Hand-written SIMD assembly kernels — no CGO, no C compiler
//   - Binary serialization ~195x faster than JSON
//   - O(n log k) top-K via min-heap
//   - Multi-tenancy with automatic LRU eviction
//   - Zero external dependencies
package nanovectordb
