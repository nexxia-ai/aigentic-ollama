//go:build integration

// run this with: go test -v -tags=integration -run ^TestOllama_ModelSuite

package ollama

import (
	"context"
	"testing"

	"github.com/nexxia-ai/aigentic/ai"
)

// TestOllama_StandardSuite runs the standard test suite against the Ollama implementation
func TestOllama_ModelSuite(t *testing.T) {
	suite := ai.ModelTestSuite{
		NewModel: func() *ai.Model {
			m := NewModel("qwen3:4b", "")
			// m.RecordFilename = "ollama_test_data.json"
			return m
		},
		SkipTests: []string{"ProcessImage"},
		Name:      "Ollama",
	}
	ai.RunModelTestSuite(t, suite)
}

func TestOllama_ProcessImage(t *testing.T) {
	model := NewModel("gemma3", "")
	ai.TestProcessImage(t, model)
}

func TestQwen_ThinkingTag(t *testing.T) {
	model := NewModel("qwen3:4b", "")
	model.WithParameter("think", true)

	ctx := context.Background()
	messages := []ai.Message{
		ai.UserMessage{Role: ai.UserRole, Content: "What is 2 + 2? Think step by step."},
	}

	response, err := model.Call(ctx, messages, []ai.Tool{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	t.Logf("Response Role: %s", response.Role)
	t.Logf("Response Content: '%s'", response.Content)
	t.Logf("Response Content Length: %d", len(response.Content))
	t.Logf("Response Think: '%s'", response.Think)
	t.Logf("Response Think Length: %d", len(response.Think))
	t.Logf("Response ToolCalls: %d", len(response.ToolCalls))

	if response.Think == "" {
		t.Error("Expected thinking content to be captured from Ollama's 'thinking' field, but it was empty")
	} else {
		t.Logf("Thinking content successfully captured: %s", response.Think)
	}

	if response.Content == "" && response.Think == "" {
		t.Error("Both content and thinking are empty - expected at least one to have content")
	}
}
