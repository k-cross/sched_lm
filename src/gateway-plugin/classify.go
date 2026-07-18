package class_aware_reliability

import (
	"hash/fnv"
	"strconv"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// Request classes, mirroring classify() in src/bench/sim/policies.py.
const (
	ClassTool    = "tool"
	ClassRAG     = "rag"
	ClassOneshot = "oneshot"
)

// ragUserChars separates a RAG query (a retrieved document embedded in the user message)
// from a session opener (a short question). The Python spec thresholds at 128 tokens on
// last_message_tokens; token counts are unavailable here, so ~4 chars/token gives 512.
const ragUserChars = 512

// UnknownTool buckets tool turns whose assistant tool_calls carry no parseable function
// name -- a single shared EWMA bucket, no worse than an unkeyed global tracker.
const UnknownTool = "unknown-tool"

// Classify infers the workload class from observables alone. Tool-session turns past the
// first carry tool-role messages (or declare tools). A lone user message is a one-shot.
// What remains is system + user: a huge user message means an embedded retrieved document
// (RAG); a short one is a session opener.
func Classify(body *requesthandling.InferenceRequestBody) string {
	if body == nil || body.ChatCompletions == nil {
		return ClassOneshot
	}
	msgs := body.ChatCompletions.Messages
	for _, m := range msgs {
		if m.Role == "tool" {
			return ClassTool
		}
	}
	if len(body.ChatCompletions.Tools) > 0 {
		return ClassTool
	}
	if len(msgs) <= 1 {
		return ClassOneshot
	}
	if len(msgs[len(msgs)-1].Content.PlainText()) >= ragUserChars {
		return ClassRAG
	}
	return ClassTool
}

// LastToolName is the name of the request's most recent tool call: the newest assistant
// message with tool_calls, last call entry, function.name. Empty when the history has no
// tool call yet (the Python spec's last_tool_name = None -- nothing learnable);
// UnknownTool when calls exist but no name can be parsed.
func LastToolName(body *requesthandling.InferenceRequestBody) string {
	if body == nil || body.ChatCompletions == nil {
		return ""
	}
	msgs := body.ChatCompletions.Messages
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role != "assistant" || len(msgs[i].ToolCalls) == 0 {
			continue
		}
		calls := msgs[i].ToolCalls
		if call, ok := calls[len(calls)-1].(map[string]any); ok {
			if fn, ok := call["function"].(map[string]any); ok {
				if name, ok := fn["name"].(string); ok && name != "" {
					return name
				}
			}
		}
		return UnknownTool
	}
	return ""
}

// SessionIDHeader is the client-supplied session identity (RFC-0001 §2). When absent, the
// key falls back to a hash of the conversation's stable opening messages.
const SessionIDHeader = "x-session-id"

// ConversationKey derives a stable identity for the conversation a request belongs to:
// the client's session header when present, else an FNV-1a hash of the system + first
// user message, which stay fixed as a session's history grows. The RFC §2 prefix-hash
// fallback scope is deferred (cross-plugin state race in v0.9.2); empty means no
// conversation link (nothing learnable).
func ConversationKey(headers map[string]string, body *requesthandling.InferenceRequestBody) string {
	if id := headers[SessionIDHeader]; id != "" {
		return id
	}
	if body == nil || body.ChatCompletions == nil || len(body.ChatCompletions.Messages) == 0 {
		return ""
	}
	h := fnv.New64a()
	remaining := 2 // system + first user message
	for _, m := range body.ChatCompletions.Messages {
		if m.Role != "system" && m.Role != "user" {
			continue
		}
		_, _ = h.Write([]byte(m.Role))
		_, _ = h.Write([]byte{0})
		_, _ = h.Write([]byte(m.Content.PlainText()))
		_, _ = h.Write([]byte{0})
		if remaining--; remaining == 0 {
			break
		}
	}
	return strconv.FormatUint(h.Sum64(), 16)
}
