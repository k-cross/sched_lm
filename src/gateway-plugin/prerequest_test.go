package class_aware_reliability

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// toolTurnBody builds a chat history shaped like a mid-session tool turn: the assistant
// called toolName and its result now sits in the history as a tool-role message.
func toolTurnBody(toolName string) *requesthandling.InferenceRequestBody {
	return &requesthandling.InferenceRequestBody{
		ChatCompletions: &requesthandling.ChatCompletionsRequest{
			Messages: []requesthandling.Message{
				{Role: "system", Content: requesthandling.Content{Raw: "You are helpful."}},
				{Role: "user", Content: requesthandling.Content{Raw: "look this up"}},
				{Role: "assistant", ToolCalls: []any{
					map[string]any{"function": map[string]any{"name": toolName}},
				}},
				{Role: "tool", Content: requesthandling.Content{Raw: "result"}},
			},
		},
	}
}

func oneshotBody() *requesthandling.InferenceRequestBody {
	return &requesthandling.InferenceRequestBody{
		ChatCompletions: &requesthandling.ChatCompletionsRequest{
			Messages: []requesthandling.Message{
				{Role: "user", Content: requesthandling.Content{Raw: "what is 2+2"}},
			},
		},
	}
}

func ragBody() *requesthandling.InferenceRequestBody {
	return &requesthandling.InferenceRequestBody{
		ChatCompletions: &requesthandling.ChatCompletionsRequest{
			Messages: []requesthandling.Message{
				{Role: "system", Content: requesthandling.Content{Raw: "You are helpful."}},
				{Role: "user", Content: requesthandling.Content{Raw: strings.Repeat("doc ", 200)}},
			},
		},
	}
}

func newRequest(body *requesthandling.InferenceRequestBody, headers map[string]string) *scheduling.InferenceRequest {
	if headers == nil {
		headers = map[string]string{}
	}
	// Directive emission is route-gated; default to the on-route so existing cases
	// exercise the emission logic. Gate tests set RouteHeader explicitly.
	if _, ok := headers[RouteHeader]; !ok {
		headers[RouteHeader] = DirectiveRoute
	}
	return &scheduling.InferenceRequest{RequestID: "req-1", Body: body, Headers: headers}
}

// testClock lets the tests drive the plugin's notion of time.
type testClock struct{ now time.Time }

func (c *testClock) Now() time.Time { return c.now }

func newTestPlugin() (*KVCachePriority, *testClock) {
	p := NewKVCachePriority("test")
	clock := &testClock{now: time.Unix(1000, 0)}
	p.now = clock.Now
	return p, clock
}

func TestPreRequestOneshot(t *testing.T) {
	p, _ := newTestPlugin()
	req := newRequest(oneshotBody(), nil)
	p.PreRequest(context.Background(), req, nil)
	assert.Equal(t, "-1", req.Headers[KVCachePriorityHeader])
}

func TestPreRequestRAGNoHeader(t *testing.T) {
	p, _ := newTestPlugin()
	req := newRequest(ragBody(), nil)
	p.PreRequest(context.Background(), req, nil)
	assert.NotContains(t, req.Headers, KVCachePriorityHeader)
}

func TestPreRequestUnseenToolNoHeader(t *testing.T) {
	// An unseen tool sits on the low-confidence default prior: no retention gamble.
	p, _ := newTestPlugin()
	req := newRequest(toolTurnBody("search"), map[string]string{SessionIDHeader: "sess1"})
	p.PreRequest(context.Background(), req, nil)
	assert.NotContains(t, req.Headers, KVCachePriorityHeader)
}

func TestPreRequestLearnsThenEmits(t *testing.T) {
	p, clock := newTestPlugin()
	headers := map[string]string{SessionIDHeader: "sess1"}

	// Turn 0: records the arrival; the default prior is not confident enough to emit.
	req := newRequest(toolTurnBody("search"), headers)
	p.PreRequest(context.Background(), req, nil)
	assert.NotContains(t, req.Headers, KVCachePriorityHeader)

	// Turn 1, 2 s later: the realized gap (2 s) seeds the index with zero variance ->
	// full confidence, short mean. Window = max(2*1.5, 2+0) = 3 s.
	clock.now = clock.now.Add(2 * time.Second)
	req = newRequest(toolTurnBody("search"), headers)
	p.PreRequest(context.Background(), req, nil)
	assert.Equal(t, "50; ttl=3s; scope=sess1", req.Headers[KVCachePriorityHeader])
}

func TestPreRequestLongGapNoHeader(t *testing.T) {
	p, clock := newTestPlugin()
	headers := map[string]string{SessionIDHeader: "sess1"}

	req := newRequest(toolTurnBody("slow-tool"), headers)
	p.PreRequest(context.Background(), req, nil)

	// A confidently long gap (10 s > shortGap 3 s) must not pin.
	clock.now = clock.now.Add(10 * time.Second)
	req = newRequest(toolTurnBody("slow-tool"), headers)
	p.PreRequest(context.Background(), req, nil)
	assert.NotContains(t, req.Headers, KVCachePriorityHeader)
}

func TestPreRequestDerivedConversationKey(t *testing.T) {
	// Without x-session-id the scope falls back to the opening-message hash, which is
	// stable across the session's turns.
	p, clock := newTestPlugin()

	req := newRequest(toolTurnBody("search"), nil)
	p.PreRequest(context.Background(), req, nil)
	assert.NotContains(t, req.Headers, KVCachePriorityHeader)

	clock.now = clock.now.Add(2 * time.Second)
	req = newRequest(toolTurnBody("search"), nil)
	p.PreRequest(context.Background(), req, nil)

	directive := req.Headers[KVCachePriorityHeader]
	require.NotEmpty(t, directive)
	assert.True(t, strings.HasPrefix(directive, "50; ttl=3s; scope="), directive)
	expectedScope := ConversationKey(map[string]string{}, toolTurnBody("search"))
	assert.Equal(t, "50; ttl=3s; scope="+expectedScope, directive)
}

func TestPreRequestClientHeaderPrecedence(t *testing.T) {
	// Router wins downward: a client asking harder than the router's ceiling is capped,
	// but a client asking for less keeps its directive.
	seedConfident := func() (*KVCachePriority, *testClock) {
		p, clock := newTestPlugin()
		headers := map[string]string{SessionIDHeader: "sess1"}
		p.PreRequest(context.Background(), newRequest(toolTurnBody("search"), headers), nil)
		clock.now = clock.now.Add(2 * time.Second)
		return p, clock
	}

	t.Run("client pins harder than ceiling -> overwritten", func(t *testing.T) {
		p, _ := seedConfident()
		req := newRequest(toolTurnBody("search"), map[string]string{
			SessionIDHeader: "sess1", KVCachePriorityHeader: "100; ttl=5m",
		})
		p.PreRequest(context.Background(), req, nil)
		assert.Equal(t, "50; ttl=3s; scope=sess1", req.Headers[KVCachePriorityHeader])
	})

	t.Run("client asks for less -> kept", func(t *testing.T) {
		p, _ := seedConfident()
		req := newRequest(toolTurnBody("search"), map[string]string{
			SessionIDHeader: "sess1", KVCachePriorityHeader: "-1",
		})
		p.PreRequest(context.Background(), req, nil)
		assert.Equal(t, "-1", req.Headers[KVCachePriorityHeader])
	})

	t.Run("no router directive -> client header untouched", func(t *testing.T) {
		p, _ := newTestPlugin()
		req := newRequest(ragBody(), map[string]string{KVCachePriorityHeader: "100; ttl=1m"})
		p.PreRequest(context.Background(), req, nil)
		assert.Equal(t, "100; ttl=1m", req.Headers[KVCachePriorityHeader])
	})
}

func TestPreRequestRouteGate(t *testing.T) {
	// Off-route requests must be fully inert: no emission (oneshots included) and no
	// gap-index training, so the off arm of an A/B run cannot leak state into the on arm.
	offRoutes := map[string]string{
		"no route header":      "",
		"prefix-affinity":      "prefix-affinity",
		"round-robin baseline": "round-robin",
	}
	for name, route := range offRoutes {
		t.Run(name, func(t *testing.T) {
			p, clock := newTestPlugin()
			headers := map[string]string{SessionIDHeader: "sess1", RouteHeader: route}

			req := newRequest(oneshotBody(), map[string]string{RouteHeader: route})
			p.PreRequest(context.Background(), req, nil)
			assert.NotContains(t, req.Headers, KVCachePriorityHeader)

			// Two tool turns 2 s apart would train a confident short gap on the
			// directive route; gated, the index must stay on the default prior.
			p.PreRequest(context.Background(), newRequest(toolTurnBody("search"), headers), nil)
			clock.now = clock.now.Add(2 * time.Second)
			req = newRequest(toolTurnBody("search"), headers)
			p.PreRequest(context.Background(), req, nil)
			assert.NotContains(t, req.Headers, KVCachePriorityHeader)

			_, variance := p.index.Predict("search")
			assert.Equal(t, DefaultPriorVar, variance, "gated traffic must not train the index")
		})
	}
}

func TestPreRequestUnknownToolBucket(t *testing.T) {
	// tool_calls without a parseable name land in the shared unknown-tool bucket and
	// still learn.
	body := func() *requesthandling.InferenceRequestBody {
		b := toolTurnBody("ignored")
		b.ChatCompletions.Messages[2].ToolCalls = []any{map[string]any{"id": "call_1"}}
		return b
	}
	p, clock := newTestPlugin()
	headers := map[string]string{SessionIDHeader: "sess1"}

	p.PreRequest(context.Background(), newRequest(body(), headers), nil)
	clock.now = clock.now.Add(2 * time.Second)
	req := newRequest(body(), headers)
	p.PreRequest(context.Background(), req, nil)
	assert.Equal(t, "50; ttl=3s; scope=sess1", req.Headers[KVCachePriorityHeader])

	mean, _ := p.index.Predict(UnknownTool)
	assert.Equal(t, 2.0, mean)
}

func TestKVCachePriorityFactory(t *testing.T) {
	raw := json.RawMessage(`{"shortGapSeconds": 5.0, "confThreshold": 0.9, "retainMargin": 2.0}`)
	instance, err := KVCachePriorityFactory("custom", plugin.StrictDecoder(raw), nil)
	require.NoError(t, err)
	p := instance.(*KVCachePriority)
	assert.Equal(t, 5.0, p.shortGap)
	assert.Equal(t, 0.9, p.confThresh)
	assert.Equal(t, 2.0, p.retainMargin)
	assert.Equal(t, plugin.TypedName{Type: KVCachePriorityType, Name: "custom"}, instance.TypedName())

	// Unknown parameters are rejected by the strict decoder.
	_, err = KVCachePriorityFactory("bad", plugin.StrictDecoder(json.RawMessage(`{"nope": 1}`)), nil)
	assert.Error(t, err)

	// No parameters -> defaults.
	instance, err = KVCachePriorityFactory("plain", nil, nil)
	require.NoError(t, err)
	assert.Equal(t, DefaultShortGapSeconds, instance.(*KVCachePriority).shortGap)
}
