package nanovectordb_test

import (
	"fmt"
	"os"
	"path/filepath"

	nanovdb "github.com/Rayen-Hamza/nanovec-go"
)

func ExampleNewNanoVectorDB() {
	dir, _ := os.MkdirTemp("", "nanovec")
	defer os.RemoveAll(dir)

	db, err := nanovdb.NewNanoVectorDB(3, filepath.Join(dir, "test.db"))
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	report, _ := db.Upsert([]nanovdb.Data{
		{
			nanovdb.FieldVector: []float32{1, 0, 0},
			nanovdb.FieldID:     "vec-x",
			"label":             "x-axis",
		},
		{
			nanovdb.FieldVector: []float32{0, 1, 0},
			nanovdb.FieldID:     "vec-y",
			"label":             "y-axis",
		},
	})
	fmt.Println("inserted:", len(report.Insert))
	fmt.Println("count:", db.Len())
	// Output:
	// inserted: 2
	// count: 2
}

func ExampleNanoVectorDB_Query() {
	dir, _ := os.MkdirTemp("", "nanovec")
	defer os.RemoveAll(dir)

	db, _ := nanovdb.NewNanoVectorDB(3, filepath.Join(dir, "test.db"))
	db.Upsert([]nanovdb.Data{
		{nanovdb.FieldVector: []float32{1, 0, 0}, nanovdb.FieldID: "x"},
		{nanovdb.FieldVector: []float32{0, 1, 0}, nanovdb.FieldID: "y"},
		{nanovdb.FieldVector: []float32{0, 0, 1}, nanovdb.FieldID: "z"},
	})

	results := db.Query([]float32{1, 0, 0}, nanovdb.QueryOption{TopK: 1})
	fmt.Println("closest:", results[0][nanovdb.FieldID])
	// Output:
	// closest: x
}

func ExampleNanoVectorDB_Query_withFilter() {
	dir, _ := os.MkdirTemp("", "nanovec")
	defer os.RemoveAll(dir)

	db, _ := nanovdb.NewNanoVectorDB(3, filepath.Join(dir, "test.db"))
	db.Upsert([]nanovdb.Data{
		{nanovdb.FieldVector: []float32{1, 0, 0}, nanovdb.FieldID: "a", "group": "red"},
		{nanovdb.FieldVector: []float32{0.9, 0.1, 0}, nanovdb.FieldID: "b", "group": "blue"},
		{nanovdb.FieldVector: []float32{0, 1, 0}, nanovdb.FieldID: "c", "group": "red"},
	})

	results := db.Query([]float32{1, 0, 0}, nanovdb.QueryOption{
		TopK: 1,
		FilterFunc: func(d nanovdb.Data) bool {
			return d["group"] == "blue"
		},
	})
	fmt.Println("closest blue:", results[0][nanovdb.FieldID])
	// Output:
	// closest blue: b
}

func ExampleNewMultiTenantNanoVDB() {
	dir, _ := os.MkdirTemp("", "nanovec-mt")
	defer os.RemoveAll(dir)

	mt, err := nanovdb.NewMultiTenantNanoVDB(3, 10, dir)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	tenantID, _ := mt.CreateTenant()
	tenant, _ := mt.GetTenant(tenantID)

	tenant.Upsert([]nanovdb.Data{
		{nanovdb.FieldVector: []float32{1, 0, 0}, "note": "hello"},
	})
	fmt.Println("tenant vectors:", tenant.Len())
	// Output:
	// tenant vectors: 1
}
