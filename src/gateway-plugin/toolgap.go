package class_aware_reliability

import (
	"container/list"
	"sync"
)

// Defaults mirroring src/bench/sim/policies.py (the offline sim is the executable spec).
const (
	DefaultGapIndexCapacity = 512
	DefaultGapAlpha         = 0.3
	// DefaultPriorMean/Var: high variance relative to the mean -> low confidence, so an
	// unseen tool does not trigger retention until it has actually been observed.
	DefaultPriorMean = 2.0
	DefaultPriorVar  = 16.0
)

// GapStat is an online estimate of a tool's re-arrival gap: EWMA mean and variance.
type GapStat struct {
	Count int
	Mean  float64
	Var   float64
}

// GapPrior is a static (mean, variance) estimate used before a tool has been observed.
type GapPrior struct {
	Mean float64
	Var  float64
}

// ToolGapIndex keeps per-tool re-arrival-gap estimates, learned from timing alone.
//
// One small table keyed by tool signature, each entry an EWMA of the observed inter-turn
// gap's mean and variance. It never sees success/failure -- only the realized gap between
// a conversation's consecutive turns. The table is capacity-bounded (an LRU); tools that
// fall off, and tools never yet seen, fall back to a static priors map, then a global
// default prior, so cost is hard-capped regardless of how many distinct tools appear.
type ToolGapIndex struct {
	mu           sync.Mutex
	capacity     int
	alpha        float64
	priors       map[string]GapPrior
	defaultPrior GapPrior
	stats        map[string]*list.Element
	order        *list.List // front = LRU, back = MRU
}

type gapEntry struct {
	tool string
	stat GapStat
}

func NewToolGapIndex(capacity int, alpha float64, priors map[string]GapPrior) *ToolGapIndex {
	if capacity <= 0 {
		capacity = DefaultGapIndexCapacity
	}
	if alpha <= 0 || alpha > 1 {
		alpha = DefaultGapAlpha
	}
	return &ToolGapIndex{
		capacity:     capacity,
		alpha:        alpha,
		priors:       priors,
		defaultPrior: GapPrior{Mean: DefaultPriorMean, Var: DefaultPriorVar},
		stats:        make(map[string]*list.Element),
		order:        list.New(),
	}
}

// Record folds one observed gap into the tool's EWMA mean and variance.
func (idx *ToolGapIndex) Record(tool string, gap float64) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	elem, ok := idx.stats[tool]
	if !ok {
		elem = idx.order.PushBack(&gapEntry{tool: tool})
		idx.stats[tool] = elem
	} else {
		idx.order.MoveToBack(elem)
	}
	stat := &elem.Value.(*gapEntry).stat
	if stat.Count == 0 {
		stat.Mean = gap
		stat.Var = 0.0
	} else {
		delta := gap - stat.Mean
		stat.Mean += idx.alpha * delta
		// EWMA of squared deviation from the (pre-update) mean -- a standard online
		// variance tracker that needs only the running mean, no second pass.
		stat.Var = (1 - idx.alpha) * (stat.Var + idx.alpha*delta*delta)
	}
	stat.Count++

	for idx.order.Len() > idx.capacity {
		lru := idx.order.Front()
		idx.order.Remove(lru)
		delete(idx.stats, lru.Value.(*gapEntry).tool)
	}
}

// Predict returns the best (mean, variance) estimate: learned stat, else prior, else
// default. An empty tool name means "no tool yet" and always yields the default prior.
func (idx *ToolGapIndex) Predict(tool string) (float64, float64) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if tool != "" {
		if elem, ok := idx.stats[tool]; ok {
			stat := elem.Value.(*gapEntry).stat
			return stat.Mean, stat.Var
		}
		if prior, ok := idx.priors[tool]; ok {
			return prior.Mean, prior.Var
		}
	}
	return idx.defaultPrior.Mean, idx.defaultPrior.Var
}

// Confidence is the inverse-variance weight in [0, 1] for the tool's estimate.
func (idx *ToolGapIndex) Confidence(tool string) float64 {
	return GapConfidence(idx.Predict(tool))
}

// GapConfidence is an inverse-variance weight in [0, 1] for a (mean, variance) gap
// estimate. Normalized against the mean so it is scale-free -- a tool whose gap varies
// little relative to its mean is trusted; a bimodal fast/slow tool is not.
func GapConfidence(mean, variance float64) float64 {
	scale := max(mean*mean, 1e-9)
	return 1.0 / (1.0 + variance/scale)
}
