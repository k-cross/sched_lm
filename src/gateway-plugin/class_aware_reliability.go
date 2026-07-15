package class_aware_reliability

import (
	"context"
	"sync"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	Name = "ClassAwareReliability"
)

var _ scheduling.ProfileHandler = &ClassAwareReliability{}

// ToolGapIndex tracks the Exponential Weighted Moving Average (EWMA) of the time between
// tool-use requests to adapt cache retention (if supported) or just track metrics.
type ToolGapIndex struct {
	mu           sync.Mutex
	lastToolTime time.Time
	ewmaMean     float64
	ewmaVar      float64
	alpha        float64
}

func NewToolGapIndex(alpha float64) *ToolGapIndex {
	return &ToolGapIndex{
		alpha: alpha,
	}
}

func (idx *ToolGapIndex) Observe() (float64, float64) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	now := time.Now()
	if !idx.lastToolTime.IsZero() {
		gap := now.Sub(idx.lastToolTime).Seconds()
		diff := gap - idx.ewmaMean
		incr := idx.alpha * diff
		idx.ewmaMean += incr
		idx.ewmaVar = (1-idx.alpha)*(idx.ewmaVar+diff*incr)
	} else {
		idx.ewmaMean = 0
		idx.ewmaVar = 0
	}
	idx.lastToolTime = now

	return idx.ewmaMean, idx.ewmaVar
}

type ClassAwareReliability struct {
	gapIndex *ToolGapIndex
}

func NewClassAwareReliability() *ClassAwareReliability {
	return &ClassAwareReliability{
		gapIndex: NewToolGapIndex(0.1),
	}
}

func (p *ClassAwareReliability) TypedName() plugin.TypedName {
	return plugin.TypedName{
		Type: "ProfileHandler",
		Name: Name,
	}
}

func (p *ClassAwareReliability) Pick(ctx context.Context, request *scheduling.InferenceRequest, profiles map[string]scheduling.SchedulerProfile, profileResults map[string]*scheduling.ProfileRunResult) map[string]scheduling.SchedulerProfile {
	// Classify the request based on InferenceRequestBody
	reqClass := "oneshot"
	
	if request.Body != nil && request.Body.ChatCompletions != nil {
		msgs := request.Body.ChatCompletions.Messages
		hasTool := false
		for _, m := range msgs {
			if m.Role == "tool" {
				hasTool = true
				break
			}
		}
		if len(request.Body.ChatCompletions.Tools) > 0 {
			hasTool = true
		}

		if hasTool {
			reqClass = "tool"
		} else if len(msgs) > 1 {
			lastMsgLen := 0
			if len(msgs) > 0 {
				lastMsgLen = len(msgs[len(msgs)-1].Content.PlainText())
			}
			// Heuristic: 128 tokens ~ 512 characters
			if lastMsgLen >= 512 {
				reqClass = "rag"
			} else {
				reqClass = "tool" // Fallback similar to Python script
			}
		}
	}

	selected := make(map[string]scheduling.SchedulerProfile)
	
	switch reqClass {
	case "tool":
		// Observe gap for tool requests
		p.gapIndex.Observe()
		// Try to pick a stateful profile, e.g. prefix-affinity
		if prof, ok := profiles["prefix-affinity"]; ok {
			selected["prefix-affinity"] = prof
		} else if prof, ok := profiles["tool-affinity"]; ok {
			selected["tool-affinity"] = prof
		}
	case "rag":
		if prof, ok := profiles["rag-affinity"]; ok {
			selected["rag-affinity"] = prof
		} else if prof, ok := profiles["prefix-affinity"]; ok {
			selected["prefix-affinity"] = prof
		}
	default:
		// oneshot goes to round-robin or least-loaded
		if prof, ok := profiles["round-robin"]; ok {
			selected["round-robin"] = prof
		} else {
			// fallback to any available
			for k, v := range profiles {
				selected[k] = v
				break
			}
		}
	}

	// If we somehow didn't select anything, fallback to first available
	if len(selected) == 0 {
		for k, v := range profiles {
			selected[k] = v
			break
		}
	}

	return selected
}

func (p *ClassAwareReliability) ProcessResults(ctx context.Context, request *scheduling.InferenceRequest, profileResults map[string]*scheduling.ProfileRunResult) (*scheduling.SchedulingResult, error) {
	// Pick the first one that succeeded
	for name, result := range profileResults {
		if result != nil {
			return &scheduling.SchedulingResult{
				ProfileResults:     profileResults,
				PrimaryProfileName: name,
			}, nil
		}
	}
	return &scheduling.SchedulingResult{
		ProfileResults: profileResults,
	}, nil
}
