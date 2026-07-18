package class_aware_reliability

import (
	"container/list"
	"sync"
	"time"
)

// GapTrackerCapacity bounds the per-conversation last-seen table used to measure
// re-arrival gaps. Memory stays O(active conversations), not O(all conversations ever) --
// single-turn RAG/one-shot keys fall out under the LRU.
const GapTrackerCapacity = 8192

// SessionTracker records when each conversation was last seen, in a bounded LRU, so the
// realized gap between a conversation's consecutive turns can be measured. Mirrors
// observe_gap in src/bench/sim/policies.py.
type SessionTracker struct {
	mu       sync.Mutex
	capacity int
	seen     map[string]*list.Element
	order    *list.List // front = LRU, back = MRU
}

type seenEntry struct {
	key  string
	last time.Time
}

func NewSessionTracker(capacity int) *SessionTracker {
	if capacity <= 0 {
		capacity = GapTrackerCapacity
	}
	return &SessionTracker{
		capacity: capacity,
		seen:     make(map[string]*list.Element),
		order:    list.New(),
	}
}

// ObserveGap returns the realized re-arrival gap since this conversation's previous turn.
//
// It records now as the conversation's latest turn and returns the gap to attribute to
// tool (the tool whose result now sits in the history). ok is false when there is nothing
// learnable: no conversation key, no prior turn, or no tool yet (turn 0 -- which still
// updates the table so the next turn's gap can be measured).
func (t *SessionTracker) ObserveGap(conversationKey, tool string, now time.Time) (gap float64, ok bool) {
	if conversationKey == "" {
		return 0, false
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	var prev time.Time
	hadPrev := false
	if elem, exists := t.seen[conversationKey]; exists {
		entry := elem.Value.(*seenEntry)
		prev, hadPrev = entry.last, true
		entry.last = now
		t.order.MoveToBack(elem)
	} else {
		t.seen[conversationKey] = t.order.PushBack(&seenEntry{key: conversationKey, last: now})
	}
	for t.order.Len() > t.capacity {
		lru := t.order.Front()
		t.order.Remove(lru)
		delete(t.seen, lru.Value.(*seenEntry).key)
	}
	if !hadPrev || tool == "" {
		return 0, false
	}
	return now.Sub(prev).Seconds(), true
}
