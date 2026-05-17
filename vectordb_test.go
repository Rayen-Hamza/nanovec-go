package nanovectordb_test

import (
	"math/rand"
	"path/filepath"
	"testing"

	nanovdb "github.com/Rayen-Hamza/nanovec-go"
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
	f := filepath.Join(t.TempDir(), "test_upsert.json")

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
	report, err := vdb.Upsert(datas)
	if err != nil {
		t.Fatal(err)
	}
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
	f := filepath.Join(t.TempDir(), "test_update.json")

	vdb, err := nanovdb.NewNanoVectorDB(dim, f)
	if err != nil {
		t.Fatal(err)
	}
	id := "fixed-id"
	d := nanovdb.Data{nanovdb.FieldID: id, nanovdb.FieldVector: randVec(dim)}
	r1, err := vdb.Upsert([]nanovdb.Data{d})
	if err != nil {
		t.Fatal(err)
	}
	if len(r1.Insert) != 1 {
		t.Fatal("expected insert")
	}
	r2, err := vdb.Upsert([]nanovdb.Data{d})
	if err != nil {
		t.Fatal(err)
	}
	if len(r2.Update) != 1 {
		t.Fatal("expected update")
	}
	if vdb.Len() != 1 {
		t.Fatalf("expected 1, got %d", vdb.Len())
	}
}

func TestDelete(t *testing.T) {
	const dim = 16
	f := filepath.Join(t.TempDir(), "test_delete.json")

	vdb, err := nanovdb.NewNanoVectorDB(dim, f)
	if err != nil {
		t.Fatal(err)
	}
	datas := []nanovdb.Data{
		{nanovdb.FieldID: "a", nanovdb.FieldVector: randVec(dim)},
		{nanovdb.FieldID: "b", nanovdb.FieldVector: randVec(dim)},
		{nanovdb.FieldID: "c", nanovdb.FieldVector: randVec(dim)},
	}
	if _, err := vdb.Upsert(datas); err != nil {
		t.Fatal(err)
	}
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
	f := filepath.Join(t.TempDir(), "test_persist.json")

	vdb, err := nanovdb.NewNanoVectorDB(dim, f)
	if err != nil {
		t.Fatal(err)
	}
	datas := make([]nanovdb.Data, 50)
	for i := range datas {
		datas[i] = nanovdb.Data{
			nanovdb.FieldVector: randVec(dim),
			"x":                 i,
		}
	}
	if _, err := vdb.Upsert(datas); err != nil {
		t.Fatal(err)
	}
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
	f := filepath.Join(t.TempDir(), "test_filter.json")

	vdb, err := nanovdb.NewNanoVectorDB(dim, f)
	if err != nil {
		t.Fatal(err)
	}
	datas := make([]nanovdb.Data, 100)
	for i := range datas {
		datas[i] = nanovdb.Data{
			nanovdb.FieldVector: randVec(dim),
			"group":             i % 2,
		}
	}
	if _, err := vdb.Upsert(datas); err != nil {
		t.Fatal(err)
	}

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
// Upsert validation tests
// ──────────────────────────────────────────────

func TestUpsertValidation(t *testing.T) {
	const dim = 16
	f := filepath.Join(t.TempDir(), "test_validation.json")
	vdb, err := nanovdb.NewNanoVectorDB(dim, f)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("missing vector", func(t *testing.T) {
		_, err := vdb.Upsert([]nanovdb.Data{{"foo": "bar"}})
		if err == nil {
			t.Fatal("expected error for missing vector")
		}
	})

	t.Run("wrong vector type", func(t *testing.T) {
		_, err := vdb.Upsert([]nanovdb.Data{{nanovdb.FieldVector: []float64{1, 2, 3}}})
		if err == nil {
			t.Fatal("expected error for wrong vector type")
		}
	})

	t.Run("wrong dimension", func(t *testing.T) {
		_, err := vdb.Upsert([]nanovdb.Data{{nanovdb.FieldVector: []float32{1, 2, 3}}})
		if err == nil {
			t.Fatal("expected error for wrong dimension")
		}
	})

	t.Run("wrong ID type", func(t *testing.T) {
		_, err := vdb.Upsert([]nanovdb.Data{{
			nanovdb.FieldVector: randVec(dim),
			nanovdb.FieldID:     12345,
		}})
		if err == nil {
			t.Fatal("expected error for wrong ID type")
		}
	})
}

// ──────────────────────────────────────────────
// MultiTenantNanoVDB tests
// ──────────────────────────────────────────────

func TestMultiTenantBasic(t *testing.T) {
	dir := t.TempDir()
	mt, err := nanovdb.NewMultiTenantNanoVDB(16, 10, dir)
	if err != nil {
		t.Fatal(err)
	}

	id, err := mt.CreateTenant()
	if err != nil {
		t.Fatal(err)
	}
	if !mt.ContainsTenant(id) {
		t.Fatal("tenant should exist after creation")
	}

	tenant, err := mt.GetTenant(id)
	if err != nil {
		t.Fatal(err)
	}
	_, err = tenant.Upsert([]nanovdb.Data{{nanovdb.FieldVector: randVec(16)}})
	if err != nil {
		t.Fatal(err)
	}
	if tenant.Len() != 1 {
		t.Fatalf("expected 1 vector, got %d", tenant.Len())
	}
}

func TestMultiTenantDelete(t *testing.T) {
	dir := t.TempDir()
	mt, err := nanovdb.NewMultiTenantNanoVDB(16, 10, dir)
	if err != nil {
		t.Fatal(err)
	}

	id, err := mt.CreateTenant()
	if err != nil {
		t.Fatal(err)
	}
	mt.DeleteTenant(id)
	if mt.ContainsTenant(id) {
		t.Fatal("tenant should not exist after deletion")
	}
}

func TestMultiTenantSaveAndReload(t *testing.T) {
	const dim = 16
	dir := t.TempDir()

	mt, err := nanovdb.NewMultiTenantNanoVDB(dim, 10, dir)
	if err != nil {
		t.Fatal(err)
	}
	id, err := mt.CreateTenant()
	if err != nil {
		t.Fatal(err)
	}
	tenant, err := mt.GetTenant(id)
	if err != nil {
		t.Fatal(err)
	}
	_, err = tenant.Upsert([]nanovdb.Data{{nanovdb.FieldVector: randVec(dim), nanovdb.FieldID: "v1"}})
	if err != nil {
		t.Fatal(err)
	}
	if err := mt.Save(); err != nil {
		t.Fatal(err)
	}

	mt2, err := nanovdb.NewMultiTenantNanoVDB(dim, 10, dir)
	if err != nil {
		t.Fatal(err)
	}
	tenant2, err := mt2.GetTenant(id)
	if err != nil {
		t.Fatal(err)
	}
	if tenant2.Len() != 1 {
		t.Fatalf("expected 1 vector after reload, got %d", tenant2.Len())
	}
}

func TestMultiTenantLRUEviction(t *testing.T) {
	const dim = 16
	dir := t.TempDir()

	mt, err := nanovdb.NewMultiTenantNanoVDB(dim, 2, dir)
	if err != nil {
		t.Fatal(err)
	}

	id1, err := mt.CreateTenant()
	if err != nil {
		t.Fatal(err)
	}
	tenant1, _ := mt.GetTenant(id1)
	_, err = tenant1.Upsert([]nanovdb.Data{{nanovdb.FieldVector: randVec(dim), nanovdb.FieldID: "t1v1"}})
	if err != nil {
		t.Fatal(err)
	}

	id2, err := mt.CreateTenant()
	if err != nil {
		t.Fatal(err)
	}

	// Touch id1 so it's most-recently-used
	mt.GetTenant(id1)

	// Creating id3 should evict id2 (LRU), not id1
	id3, err := mt.CreateTenant()
	if err != nil {
		t.Fatal(err)
	}

	// id1 should still be accessible (was touched)
	t1, err := mt.GetTenant(id1)
	if err != nil {
		t.Fatal("id1 should be accessible after LRU touch")
	}
	if t1.Len() != 1 {
		t.Fatalf("id1 should still have 1 vector, got %d", t1.Len())
	}

	// id2 should be loadable from disk
	_, err = mt.GetTenant(id2)
	if err != nil {
		t.Fatal("id2 should be loadable from disk after eviction")
	}

	// id3 should be accessible
	_, err = mt.GetTenant(id3)
	if err != nil {
		t.Fatal("id3 should be accessible")
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

	datas := make([]nanovdb.Data, n)
	for i := range datas {
		datas[i] = nanovdb.Data{nanovdb.FieldVector: randVec(dim)}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f := filepath.Join(b.TempDir(), "bench_upsert.json")
		vdb, err := nanovdb.NewNanoVectorDB(dim, f)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := vdb.Upsert(datas); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSave100k(b *testing.B) {
	const (
		n   = 100_000
		dim = 1024
	)
	f := filepath.Join(b.TempDir(), "bench_save.json")

	vdb, err := nanovdb.NewNanoVectorDB(dim, f)
	if err != nil {
		b.Fatal(err)
	}
	datas := make([]nanovdb.Data, n)
	for i := range datas {
		datas[i] = nanovdb.Data{nanovdb.FieldVector: randVec(dim)}
	}
	if _, err := vdb.Upsert(datas); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := vdb.Save(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLoad100k(b *testing.B) {
	const (
		n   = 100_000
		dim = 1024
	)
	f := filepath.Join(b.TempDir(), "bench_load.json")

	vdb, err := nanovdb.NewNanoVectorDB(dim, f)
	if err != nil {
		b.Fatal(err)
	}
	datas := make([]nanovdb.Data, n)
	for i := range datas {
		datas[i] = nanovdb.Data{nanovdb.FieldVector: randVec(dim)}
	}
	if _, err := vdb.Upsert(datas); err != nil {
		b.Fatal(err)
	}
	if err := vdb.Save(); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := nanovdb.NewNanoVectorDB(dim, f)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQuery100k(b *testing.B) {
	const (
		n   = 100_000
		dim = 1024
	)
	f := filepath.Join(b.TempDir(), "bench_query.json")

	vdb, err := nanovdb.NewNanoVectorDB(dim, f)
	if err != nil {
		b.Fatal(err)
	}
	datas := make([]nanovdb.Data, n)
	for i := range datas {
		datas[i] = nanovdb.Data{nanovdb.FieldVector: randVec(dim)}
	}
	if _, err := vdb.Upsert(datas); err != nil {
		b.Fatal(err)
	}
	q := randVec(dim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vdb.Query(q, nanovdb.QueryOption{TopK: 10})
	}
}
