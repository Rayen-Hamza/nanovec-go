package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"

	nanovdb "github.com/Rayen-Hamza/nanovec-go"
)

func main() {
	dir, err := os.MkdirTemp("", "tenants")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)

	mt, err := nanovdb.NewMultiTenantNanoVDB(64, 10, dir)
	if err != nil {
		log.Fatal(err)
	}

	// Create two tenants
	rng := rand.New(rand.NewSource(42))
	for t := 0; t < 2; t++ {
		id, err := mt.CreateTenant()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Created tenant: %s\n", id)

		tenant, _ := mt.GetTenant(id)

		batch := make([]nanovdb.Data, 50)
		for i := range batch {
			vec := make([]float32, 64)
			for j := range vec {
				vec[j] = rng.Float32()*2 - 1
			}
			batch[i] = nanovdb.Data{
				nanovdb.FieldVector: vec,
				"source":            fmt.Sprintf("tenant-%d", t),
			}
		}
		tenant.Upsert(batch)
		fmt.Printf("  Inserted %d vectors\n", tenant.Len())
	}

	// Save all tenants to disk
	if err := mt.Save(); err != nil {
		log.Fatal(err)
	}
	fmt.Println("\nAll tenants saved to disk")
}
