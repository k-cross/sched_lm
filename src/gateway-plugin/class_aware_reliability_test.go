package class_aware_reliability

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

type mockProfile struct{}

func (m *mockProfile) Run(ctx context.Context, request *scheduling.InferenceRequest, candidateEndpoints []scheduling.Endpoint) (*scheduling.ProfileRunResult, error) {
	return &scheduling.ProfileRunResult{}, nil
}

func TestClassAwareReliability_Pick(t *testing.T) {
	plugin := NewClassAwareReliability()

	profiles := map[string]scheduling.SchedulerProfile{
		"prefix-affinity": &mockProfile{},
		"rag-affinity":    &mockProfile{},
		"round-robin":     &mockProfile{},
	}

	tests := []struct {
		name          string
		body          *requesthandling.InferenceRequestBody
		expectedRoute string
	}{
		{
			name: "oneshot request with empty chat",
			body: &requesthandling.InferenceRequestBody{
				ChatCompletions: &requesthandling.ChatCompletionsRequest{
					Messages: []requesthandling.Message{},
				},
			},
			expectedRoute: "round-robin",
		},
		{
			name: "tool request with role tool message",
			body: &requesthandling.InferenceRequestBody{
				ChatCompletions: &requesthandling.ChatCompletionsRequest{
					Messages: []requesthandling.Message{
						{Role: "user", Content: requesthandling.Content{Raw: "hello"}},
						{Role: "tool", Content: requesthandling.Content{Raw: "result"}},
					},
				},
			},
			expectedRoute: "prefix-affinity",
		},
		{
			name: "tool request with tools defined",
			body: &requesthandling.InferenceRequestBody{
				ChatCompletions: &requesthandling.ChatCompletionsRequest{
					Messages: []requesthandling.Message{
						{Role: "user", Content: requesthandling.Content{Raw: "hello"}},
					},
					Tools: []any{map[string]any{"type": "function"}},
				},
			},
			expectedRoute: "prefix-affinity",
		},
		{
			name: "rag request with long message",
			body: &requesthandling.InferenceRequestBody{
				ChatCompletions: &requesthandling.ChatCompletionsRequest{
					Messages: []requesthandling.Message{
						{Role: "user", Content: requesthandling.Content{Raw: "system prompt"}},
						{Role: "user", Content: requesthandling.Content{Raw: string(make([]byte, 512))}}, // > 512 chars
					},
				},
			},
			expectedRoute: "rag-affinity",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &scheduling.InferenceRequest{
				Body: tt.body,
			}
			selected := plugin.Pick(context.Background(), req, profiles, nil)
			assert.Len(t, selected, 1)
			for routeName := range selected {
				assert.Equal(t, tt.expectedRoute, routeName)
			}
		})
	}
}
