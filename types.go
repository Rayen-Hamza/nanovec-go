package nanovectordb

// Field name constants for reserved keys in Data maps.
const (
	FieldID      = "__id__"
	FieldVector  = "__vector__"
	FieldMetrics = "__metrics__"
)

// Data is a row: arbitrary string-keyed fields plus FieldVector ([]float32).
type Data map[string]any

// UpsertReport describes which IDs were inserted vs updated.
type UpsertReport struct {
	Update []string `json:"update"`
	Insert []string `json:"insert"`
}

// QueryOption configures a Query call.
type QueryOption struct {
	TopK                int
	BetterThanThreshold *float32
	FilterFunc          func(Data) bool
}
