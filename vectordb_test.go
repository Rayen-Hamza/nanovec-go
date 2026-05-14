package nanovectordb_test

import (
	"fmt"
	"math/rand"
	"os"
	"testing"

	nanovdb "nano-vectordb"
)

func randVec(dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rand.Float32()
	}
	return v
}

func TestUpsertAndQuery(t *testing.T) {
	const dim = 64
	f := "/tmp/test_upsert.json"
	defer os.Remove(f)

	vdb, err := nanovdb.NewNanoVectorDB(dim, f)
	if err != nil {
		t.Fatal(err)
	}

	datas := make([]nanovdb.Data, 200)
	for i := range datas {
		datas[i] = nanovdb.Data{
			nanovdb.FieldVector: randVec(dim),
			"idx":               i,
		}
	}
	report := vdb.Upsert(datas)
	if len(report.Insert) != 200 {
		t.Fatalf("expected 200 inserts, got %d", len(report.Insert))
	}
	if vdb.Len() != 200 {
		t.Fatalf("expected len 200, got %d", vdb.Len())
	}

	results := vdb.Query(randVec(dim), nanovdb.QueryOption{TopK: 5})
	if len(results) != 5 {
		t.Fatalf("expected 5 results, got %d", len(results))
	}
	// Scores must be descending
	for i := 1; i < len(results); i++ {
		prev := results[i-1][nanovdb.FieldMetrics].(float32)
		cur := results[i][nanovdb.FieldMetrics].(float32)
		if prev < cur {
			t.Fatalf("scores not descending at index %d: %f < %f", i, prev, cur)
		}
	}
}

func TestUpsertUpdate(t *testing.T) {
	const dim = 16
	f := "/tmp/test_update.json"
	defer os.Remove(f)

	vdb, _ := nanovdb.NewNanoVectorDB(dim, f)
	id := "fixed-id"
	d := nanovdb.Data{nanovdb.FieldID: id, nanovdb.FieldVector: randVec(dim)}
	r1 := vdb.Upsert([]nanovdb.Data{d})
	if len(r1.Insert) != 1 {
		t.Fatal("expected insert")
	}
	r2 := vdb.Upsert([]nanovdb.Data{d})
	if len(r2.Update) != 1 {
		t.Fatal("expected update")
	}
	if vdb.Len() != 1 {
		t.Fatalf("expected 1, got %d", vdb.Len())
	}
}

func TestDelete(t *testing.T) {
	const dim = 16
	f := "/tmp/test_delete.json"
	defer os.Remove(f)

	vdb, _ := nanovdb.NewNanoVectorDB(dim, f)
	datas := []nanovdb.Data{
		{nanovdb.FieldID: "a", nanovdb.FieldVector: randVec(dim)},
		{nanovdb.FieldID: "b", nanovdb.FieldVector: randVec(dim)},
		{nanovdb.FieldID: "c", nanovdb.FieldVector: randVec(dim)},
	}
	vdb.Upsert(datas)
	vdb.Delete([]string{"b"})
	if vdb.Len() != 2 {
		t.Fatalf("expected 2 after delete, got %d", vdb.Len())
	}
	got := vdb.Get([]string{"b"})
	if len(got) != 0 {
		t.Fatal("deleted item should not be found")
	}
}

func TestSaveAndReload(t *testing.T) {
	const dim = 32
	f := "/tmp/test_persist.json"
	defer os.Remove(f)

	vdb, _ := nanovdb.NewNanoVectorDB(dim, f)
	datas := make([]nanovdb.Data, 50)
	for i := range datas {
		datas[i] = nanovdb.Data{
			nanovdb.FieldVector: randVec(dim),
			"x":                 fmt.Sprintf("v%d", i),
		}
	}
	vdb.Upsert(datas)
	if err := vdb.Save(); err != nil {
		t.Fatal(err)
	}

	vdb2, err := nanovdb.NewNanoVectorDB(dim, f)
	if err != nil {
		t.Fatal(err)
	}
	if vdb2.Len() != 50 {
		t.Fatalf("expected 50 after reload, got %d", vdb2.Len())
	}
}

func TestFilterQuery(t *testing.T) {
	const dim = 16
	f := "/tmp/test_filter.json"
	defer os.Remove(f)

	vdb, _ := nanovdb.NewNanoVectorDB(dim, f)
	datas := make([]nanovdb.Data, 100)
	for i := range datas {
		datas[i] = nanovdb.Data{
			nanovdb.FieldVector: randVec(dim),
			"group":             i % 2,
		}
	}
	vdb.Upsert(datas)

	results := vdb.Query(randVec(dim), nanovdb.QueryOption{
		TopK: 10,
		FilterFunc: func(d nanovdb.Data) bool {
			g, _ := d["group"].(float64)
			return g == 0
		},
	})
	for _, r := range results {
		g, _ := r["group"].(float64)
		if g != 0 {
			t.Fatal("filter not applied correctly")
		}
	}
}

// ──────────────────────────────────────────────
// Benchmarks
// ──────────────────────────────────────────────

func BenchmarkUpsert100k(b *testing.B) {
	const (
		n   = 100_000
		dim = 1024
	)
	f := "/tmp/bench_upsert.json"
	defer os.Remove(f)

	datas := make([]nanovdb.Data, n)
	for i := range datas {
		datas[i] = nanovdb.Data{nanovdb.FieldVector: randVec(dim)}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		os.Remove(f)
		vdb, _ := nanovdb.NewNanoVectorDB(dim, f)
		vdb.Upsert(datas)
	}
}

func BenchmarkQuery100k(b *testing.B) {
	const (
		n   = 100_000
		dim = 1024
	)
	f := "/tmp/bench_query.json"
	defer os.Remove(f)

	vdb, _ := nanovdb.NewNanoVectorDB(dim, f)
	datas := make([]nanovdb.Data, n)
	for i := range datas {
		datas[i] = nanovdb.Data{nanovdb.FieldVector: randVec(dim)}
	}
	vdb.Upsert(datas)
	q := randVec(dim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vdb.Query(q, nanovdb.QueryOption{TopK: 10})
	}
}
