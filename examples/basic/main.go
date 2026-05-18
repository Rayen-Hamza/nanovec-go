package main

import (
	"fmt"
	"log"
	"math/rand"

	nanovdb "github.com/Rayen-Hamza/nanovec-go"
)

func main() {
	db, err := nanovdb.NewNanoVectorDB(128, "vectors.db")
	if err != nil {
		log.Fatal(err)
	}

	// Insert 100 vectors with metadata
	batch := make([]nanovdb.Data, 100)
	rng := rand.New(rand.NewSource(42))
	for i := range batch {
		vec := make([]float32, 128)
		for j := range vec {
			vec[j] = rng.Float32()*2 - 1
		}
		batch[i] = nanovdb.Data{
			nanovdb.FieldVector: vec,
			"label":             fmt.Sprintf("doc-%d", i),
			"group":             fmt.Sprintf("group-%d", i%3),
		}
	}

	report, err := db.Upsert(batch)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Inserted %d vectors\n", len(report.Insert))

	// Query top-5 nearest neighbors
	query := make([]float32, 128)
	for i := range query {
		query[i] = rng.Float32()*2 - 1
	}

	results := db.Query(query, nanovdb.QueryOption{TopK: 5})
	fmt.Println("\nTop-5 results:")
	for _, r := range results {
		fmt.Printf("  %s  score=%.4f  group=%s\n",
			r[nanovdb.FieldID], r[nanovdb.FieldMetrics], r["group"])
	}

	// Query with filter
	results = db.Query(query, nanovdb.QueryOption{
		TopK: 3,
		FilterFunc: func(d nanovdb.Data) bool {
			return d["group"] == "group-0"
		},
	})
	fmt.Println("\nTop-3 in group-0:")
	for _, r := range results {
		fmt.Printf("  %s  score=%.4f\n", r[nanovdb.FieldID], r[nanovdb.FieldMetrics])
	}

	// Save to disk
	if err := db.Save(); err != nil {
		log.Fatal(err)
	}
	fmt.Println("\nSaved to vectors.db")
}
