package nanovectordb

import (
	"crypto/rand"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// MultiTenantNanoVDB manages multiple NanoVectorDB instances with LRU eviction.
type MultiTenantNanoVDB struct {
	EmbeddingDim int
	MaxCapacity  int
	StorageDir   string

	mu       sync.Mutex
	tenants  map[string]*NanoVectorDB
	lruQueue []string
}

// NewMultiTenantNanoVDB creates a multi-tenant manager.
func NewMultiTenantNanoVDB(embeddingDim, maxCapacity int, storageDir string) (*MultiTenantNanoVDB, error) {
	if maxCapacity < 1 {
		return nil, fmt.Errorf("maxCapacity must be >= 1")
	}
	if err := os.MkdirAll(storageDir, 0o755); err != nil {
		return nil, err
	}
	return &MultiTenantNanoVDB{
		EmbeddingDim: embeddingDim,
		MaxCapacity:  maxCapacity,
		StorageDir:   storageDir,
		tenants:      make(map[string]*NanoVectorDB),
	}, nil
}

func (m *MultiTenantNanoVDB) jsonFile(id string) string {
	return filepath.Join(m.StorageDir, fmt.Sprintf("nanovdb_%s.json", id))
}

// ContainsTenant checks if a tenant exists in memory or on disk.
func (m *MultiTenantNanoVDB) ContainsTenant(id string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.tenants[id]; ok {
		return true
	}
	_, err := os.Stat(m.jsonFile(id))
	return err == nil
}

func (m *MultiTenantNanoVDB) touchLRU(id string) {
	for i, qid := range m.lruQueue {
		if qid == id {
			m.lruQueue = append(m.lruQueue[:i], m.lruQueue[i+1:]...)
			break
		}
	}
	m.lruQueue = append(m.lruQueue, id)
}

func (m *MultiTenantNanoVDB) evict() error {
	if len(m.tenants) < m.MaxCapacity {
		return nil
	}
	oldest := m.lruQueue[0]
	m.lruQueue = m.lruQueue[1:]
	if vdb, ok := m.tenants[oldest]; ok {
		if err := vdb.Save(); err != nil {
			return err
		}
		delete(m.tenants, oldest)
	}
	return nil
}

// CreateTenant creates a new tenant and returns its ID.
func (m *MultiTenantNanoVDB) CreateTenant() (string, error) {
	id := newUUID()
	m.mu.Lock()
	defer m.mu.Unlock()
	if err := m.evict(); err != nil {
		return "", err
	}
	vdb, err := NewNanoVectorDB(m.EmbeddingDim, m.jsonFile(id))
	if err != nil {
		return "", err
	}
	m.tenants[id] = vdb
	m.lruQueue = append(m.lruQueue, id)
	return id, nil
}

// GetTenant returns a tenant's DB, loading from disk if needed.
func (m *MultiTenantNanoVDB) GetTenant(id string) (*NanoVectorDB, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if vdb, ok := m.tenants[id]; ok {
		m.touchLRU(id)
		return vdb, nil
	}
	if _, err := os.Stat(m.jsonFile(id)); err != nil {
		return nil, fmt.Errorf("tenant %s not found", id)
	}
	if err := m.evict(); err != nil {
		return nil, err
	}
	vdb, err := NewNanoVectorDB(m.EmbeddingDim, m.jsonFile(id))
	if err != nil {
		return nil, err
	}
	m.tenants[id] = vdb
	m.lruQueue = append(m.lruQueue, id)
	return vdb, nil
}

// DeleteTenant removes a tenant from memory and disk.
func (m *MultiTenantNanoVDB) DeleteTenant(id string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.tenants, id)
	for i, qid := range m.lruQueue {
		if qid == id {
			m.lruQueue = append(m.lruQueue[:i], m.lruQueue[i+1:]...)
			break
		}
	}
	_ = os.Remove(m.jsonFile(id))
}

// Save flushes all in-memory tenants to disk.
func (m *MultiTenantNanoVDB) Save() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, vdb := range m.tenants {
		if err := vdb.Save(); err != nil {
			return err
		}
	}
	return nil
}

func newUUID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:])
}
