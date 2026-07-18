package class_aware_reliability

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// KVCachePriorityHeader is the compact router-path transport (RFC-0001 §2):
//
//	x-kv-cache-priority: <int>[; ttl=<Go duration>][; scope=<id>]
//
// Values must stay in sync with the backend grammar in the llm-d-inference-sim fork's
// pkg/retention (a separate Go module, hence not imported).
const (
	KVCachePriorityHeader = "x-kv-cache-priority"
	EvictFirstPriority    = -1
	HighPriority          = 50
)

// Retention decision defaults, mirroring ClassAwareReliability in
// src/bench/sim/policies.py.
const (
	DefaultShortGapSeconds = 3.0
	DefaultConfThreshold   = 0.5
	DefaultRetainMargin    = 1.5
)

// KVCachePriorityType is the plugin type string declared in EndpointPickerConfig.
const KVCachePriorityType = "kv-cache-priority"

var _ requestcontrol.PreRequest = &KVCachePriority{}

// KVCachePriority is the RFC-0001 §5 router-side retention hinter: a PreRequest plugin
// that classifies the request, learns per-tool re-arrival gaps from timing alone, and
// injects x-kv-cache-priority after scheduling. One-shots -- no future reuse, pure cache
// pollution -- are marked evict-first. A tool-session turn whose most recent tool call
// confidently predicts a short return gets a HIGH lease sized to the predicted gap, so
// the prefix survives eviction until the session comes back. A long predicted gap, or an
// unpredictable (high-variance) or unseen tool, adds no retention: the router does not
// gamble cache on an unreliable prediction. Directives are hints -- they never affect
// correctness -- and the backend's lease policy caps every TTL.
type KVCachePriority struct {
	name         string
	index        *ToolGapIndex
	sessions     *SessionTracker
	shortGap     float64
	confThresh   float64
	retainMargin float64
	now          func() time.Time
}

// kvCachePriorityParams are the optional EndpointPickerConfig parameters.
type kvCachePriorityParams struct {
	ShortGapSeconds *float64 `json:"shortGapSeconds"`
	ConfThreshold   *float64 `json:"confThreshold"`
	RetainMargin    *float64 `json:"retainMargin"`
}

func NewKVCachePriority(name string) *KVCachePriority {
	return &KVCachePriority{
		name:         name,
		index:        NewToolGapIndex(DefaultGapIndexCapacity, DefaultGapAlpha, nil),
		sessions:     NewSessionTracker(GapTrackerCapacity),
		shortGap:     DefaultShortGapSeconds,
		confThresh:   DefaultConfThreshold,
		retainMargin: DefaultRetainMargin,
		now:          time.Now,
	}
}

// KVCachePriorityFactory instantiates the plugin from an EndpointPickerConfig entry.
func KVCachePriorityFactory(name string, parameters *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	p := NewKVCachePriority(name)
	if parameters != nil {
		var params kvCachePriorityParams
		if err := parameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse %s parameters: %w", KVCachePriorityType, err)
		}
		if params.ShortGapSeconds != nil {
			p.shortGap = *params.ShortGapSeconds
		}
		if params.ConfThreshold != nil {
			p.confThresh = *params.ConfThreshold
		}
		if params.RetainMargin != nil {
			p.retainMargin = *params.RetainMargin
		}
	}
	return p, nil
}

func (p *KVCachePriority) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: KVCachePriorityType, Name: p.name}
}

// PreRequest runs after scheduling and before header serialization; mutating
// request.Headers propagates to the backend via the extProc HeaderMutation.
func (p *KVCachePriority) PreRequest(ctx context.Context, request *scheduling.InferenceRequest, _ *scheduling.SchedulingResult) {
	if request == nil || request.Headers == nil {
		return
	}
	directive, ok := p.decide(request)
	if !ok {
		return
	}
	// Precedence (RFC-0001 §2): the router wins downward -- its value caps the effective
	// priority -- but never raises a client directive that already asks for less.
	if existing, present := request.Headers[KVCachePriorityHeader]; present {
		if clientPriority, parsed := leadingPriority(existing); parsed && clientPriority <= priorityOf(directive) {
			return
		}
	}
	request.Headers[KVCachePriorityHeader] = directive
	log.FromContext(ctx).V(1).Info("kv-cache-priority directive emitted",
		"requestID", request.RequestID, "directive", directive)
}

// decide computes the directive for this request, or ok=false for no hint (normal LRU).
func (p *KVCachePriority) decide(request *scheduling.InferenceRequest) (string, bool) {
	class := Classify(request.Body)
	if class == ClassOneshot {
		return strconv.Itoa(EvictFirstPriority), true
	}
	if class != ClassTool {
		return "", false
	}

	now := p.now()
	tool := LastToolName(request.Body)
	scope := ConversationKey(request.Headers, request.Body)
	if gap, ok := p.sessions.ObserveGap(scope, tool, now); ok {
		p.index.Record(tool, gap)
	}

	mean, variance := p.index.Predict(tool)
	if GapConfidence(mean, variance) < p.confThresh || mean > p.shortGap {
		return "", false
	}
	// Keep the prefix warm at least mean * margin, but extend by one standard deviation
	// when the (confident-but-nonzero) spread pushes likely returns past that.
	window := max(mean*p.retainMargin, mean+math.Sqrt(variance))
	ttl := time.Duration(window * float64(time.Second)).Round(time.Millisecond)
	directive := fmt.Sprintf("%d; ttl=%s", HighPriority, ttl)
	if scope != "" {
		directive += "; scope=" + scope
	}
	return directive, true
}

// priorityOf extracts the leading priority of a directive this plugin built.
func priorityOf(directive string) int {
	priority, _ := leadingPriority(directive)
	return priority
}

// leadingPriority parses the leading integer of an x-kv-cache-priority value.
func leadingPriority(value string) (int, bool) {
	head, _, _ := strings.Cut(value, ";")
	priority, err := strconv.Atoi(strings.TrimSpace(head))
	return priority, err == nil
}
