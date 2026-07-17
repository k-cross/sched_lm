package class_aware_reliability

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestToolGapIndexPriorsBeforeObservation(t *testing.T) {
	idx := NewToolGapIndex(4, 0.3, map[string]GapPrior{"search": {Mean: 1.0, Var: 0.25}})

	mean, variance := idx.Predict("search")
	assert.Equal(t, 1.0, mean)
	assert.Equal(t, 0.25, variance)

	// Unseen tool without a prior, and "no tool yet", both fall to the default prior.
	for _, tool := range []string{"never-seen", ""} {
		mean, variance = idx.Predict(tool)
		assert.Equal(t, DefaultPriorMean, mean)
		assert.Equal(t, DefaultPriorVar, variance)
	}

	// A learned stat overrides the prior.
	idx.Record("search", 3.0)
	mean, variance = idx.Predict("search")
	assert.Equal(t, 3.0, mean)
	assert.Equal(t, 0.0, variance)
}

func TestToolGapIndexEWMA(t *testing.T) {
	idx := NewToolGapIndex(4, 0.3, nil)

	// First sample seeds the estimate exactly.
	idx.Record("search", 2.0)
	mean, variance := idx.Predict("search")
	assert.Equal(t, 2.0, mean)
	assert.Equal(t, 0.0, variance)

	// Second sample follows the spec update: delta = 4-2 = 2;
	// mean = 2 + 0.3*2 = 2.6; var = 0.7*(0 + 0.3*4) = 0.84.
	idx.Record("search", 4.0)
	mean, variance = idx.Predict("search")
	assert.InDelta(t, 2.6, mean, 1e-12)
	assert.InDelta(t, 0.84, variance, 1e-12)

	// A steady stream of identical gaps converges the mean and squeezes the variance.
	for range 50 {
		idx.Record("steady", 1.5)
	}
	mean, variance = idx.Predict("steady")
	assert.InDelta(t, 1.5, mean, 1e-9)
	assert.Less(t, variance, 1e-9)
	assert.Greater(t, GapConfidence(mean, variance), 0.99)
}

func TestToolGapIndexLRUEviction(t *testing.T) {
	idx := NewToolGapIndex(2, 0.3, nil)

	idx.Record("a", 1.0)
	idx.Record("b", 2.0)
	idx.Record("a", 1.0) // refresh "a" as most-recently-used
	idx.Record("c", 3.0) // evicts "b", the LRU

	mean, _ := idx.Predict("a")
	assert.Equal(t, 1.0, mean)
	mean, _ = idx.Predict("c")
	assert.Equal(t, 3.0, mean)
	mean, variance := idx.Predict("b") // fell off -> default prior
	assert.Equal(t, DefaultPriorMean, mean)
	assert.Equal(t, DefaultPriorVar, variance)
}

func TestGapConfidence(t *testing.T) {
	// Zero variance -> full confidence; variance equal to mean^2 -> 0.5.
	assert.Equal(t, 1.0, GapConfidence(2.0, 0.0))
	assert.InDelta(t, 0.5, GapConfidence(2.0, 4.0), 1e-12)
	// The default prior is deliberately low-confidence (below the retention threshold).
	assert.Less(t, GapConfidence(DefaultPriorMean, DefaultPriorVar), DefaultConfThreshold)
}

func TestToolGapIndexCapacityBound(t *testing.T) {
	idx := NewToolGapIndex(8, 0.3, nil)
	for i := range 100 {
		idx.Record(fmt.Sprintf("tool-%d", i), float64(i))
	}
	assert.Equal(t, 8, idx.order.Len())
	assert.Len(t, idx.stats, 8)
}
