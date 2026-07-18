package class_aware_reliability

import (
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestSessionTrackerObserveGap(t *testing.T) {
	tracker := NewSessionTracker(8)
	t0 := time.Unix(1000, 0)

	// No conversation link -> nothing learnable, nothing recorded.
	_, ok := tracker.ObserveGap("", "search", t0)
	assert.False(t, ok)

	// Turn 0: records the arrival but has no prior turn to measure against.
	_, ok = tracker.ObserveGap("conv", "search", t0)
	assert.False(t, ok)

	// Turn 1: the realized gap since turn 0.
	gap, ok := tracker.ObserveGap("conv", "search", t0.Add(2500*time.Millisecond))
	assert.True(t, ok)
	assert.InDelta(t, 2.5, gap, 1e-9)

	// A turn with no tool yet still updates last-seen (so the next gap is measured from
	// it) but returns nothing to attribute.
	_, ok = tracker.ObserveGap("conv", "", t0.Add(4*time.Second))
	assert.False(t, ok)
	gap, ok = tracker.ObserveGap("conv", "search", t0.Add(5*time.Second))
	assert.True(t, ok)
	assert.InDelta(t, 1.0, gap, 1e-9)
}

func TestSessionTrackerCapacityBound(t *testing.T) {
	tracker := NewSessionTracker(4)
	t0 := time.Unix(1000, 0)

	for i := range 10 {
		tracker.ObserveGap(fmt.Sprintf("conv-%d", i), "search", t0)
	}
	assert.Equal(t, 4, tracker.order.Len())
	assert.Len(t, tracker.seen, 4)

	// conv-0 fell off the LRU: its next turn looks like turn 0 again.
	_, ok := tracker.ObserveGap("conv-0", "search", t0.Add(time.Second))
	assert.False(t, ok)
	// conv-9 survived: its gap is measurable.
	gap, ok := tracker.ObserveGap("conv-9", "search", t0.Add(2*time.Second))
	assert.True(t, ok)
	assert.InDelta(t, 2.0, gap, 1e-9)
}
