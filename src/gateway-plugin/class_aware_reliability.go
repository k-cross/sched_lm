package class_aware_reliability

import (
	"context"
	"encoding/json"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	Name = "ClassAwareReliability"
	// ClassAwareReliabilityType is the plugin type string for EndpointPickerConfig.
	// Registered but unwired in the phase-4 minimal config; profile-routing A/B is phase 5.
	ClassAwareReliabilityType = "class-aware-reliability"
)

var _ scheduling.ProfileHandler = &ClassAwareReliability{}

// ClassAwareReliability is a ProfileHandler that picks a scheduling profile by workload
// class (tool / rag / oneshot), classified from request observables alone. The per-tool
// gap learning and retention-directive emission live in the KVCachePriority PreRequest
// plugin; this handler only routes.
type ClassAwareReliability struct {
	name string
}

func NewClassAwareReliability() *ClassAwareReliability {
	return &ClassAwareReliability{name: Name}
}

// ClassAwareReliabilityFactory instantiates the handler from an EndpointPickerConfig entry.
func ClassAwareReliabilityFactory(name string, _ *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	handler := NewClassAwareReliability()
	if name != "" {
		handler.name = name
	}
	return handler, nil
}

func (p *ClassAwareReliability) TypedName() plugin.TypedName {
	return plugin.TypedName{
		Type: ClassAwareReliabilityType,
		Name: p.name,
	}
}

func (p *ClassAwareReliability) Pick(ctx context.Context, request *scheduling.InferenceRequest, profiles map[string]scheduling.SchedulerProfile, profileResults map[string]*scheduling.ProfileRunResult) map[string]scheduling.SchedulerProfile {
	selected := make(map[string]scheduling.SchedulerProfile)

	switch Classify(request.Body) {
	case ClassTool:
		if prof, ok := profiles["prefix-affinity"]; ok {
			selected["prefix-affinity"] = prof
		} else if prof, ok := profiles["tool-affinity"]; ok {
			selected["tool-affinity"] = prof
		}
	case ClassRAG:
		if prof, ok := profiles["rag-affinity"]; ok {
			selected["rag-affinity"] = prof
		} else if prof, ok := profiles["prefix-affinity"]; ok {
			selected["prefix-affinity"] = prof
		}
	default:
		// oneshot goes to round-robin or least-loaded
		if prof, ok := profiles["round-robin"]; ok {
			selected["round-robin"] = prof
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
