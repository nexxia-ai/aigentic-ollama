//go:build integration

// run this with: go test -tags=integration -run ^TestOllama_StandardSuite

package ollama

import (
	"os"
	"testing"

	"github.com/nexxia-ai/aigentic/ai"
	"github.com/nexxia-ai/aigentic/tests/integration"
)

// TestOllama_StandardSuite runs the standard test suite against the Ollama implementation
func TestOllama_StandardSuite(t *testing.T) {
	suite := integration.ModelTestSuite{
		NewModel: func() *ai.Model {
			m := NewModel("qwen3:1.7b", "")
			// m.RecordFilename = "ollama_test_data.json"
			return m
		},
		Name: "Ollama",
	}
	integration.RunModelTestSuite(t, suite)
}

// TestOllama_IndividualTests demonstrates how to run individual tests
func TestOllama_IndividualTests(t *testing.T) {
	model := NewModel("qwen3:1.7b", "")

	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	t.Run("GenerateSimple", func(t *testing.T) {
		integration.TestGenerateSimple(t, model)
	})

	t.Run("ProcessImage", func(t *testing.T) {
		integration.TestProcessImage(t, model)
	})

	t.Run("ProcessAttachments", func(t *testing.T) {
		integration.TestProcessAttachments(t, model)
	})

	t.Run("GenerateContentWithTools", func(t *testing.T) {
		integration.TestGenerateContentWithTools(t, model)
	})
}
