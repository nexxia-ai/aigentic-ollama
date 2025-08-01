//go:build integration

// run this with: go test -v -tags=integration -run ^TestOllama_ModelSuite

package ollama

import (
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
		Name: "Ollama",
	}
	ai.RunModelTestSuite(t, suite)
}

// TestOllama_IndividualTests demonstrates how to run individual tests
func TestOllama_IndividualTests(t *testing.T) {
	model := NewModel("qwen3:4b", "")

	t.Run("GenerateSimple", func(t *testing.T) {
		ai.TestGenerateSimple(t, model)
	})

	t.Run("ProcessImage", func(t *testing.T) {
		ai.TestProcessImage(t, model)
	})

	t.Run("ProcessAttachments", func(t *testing.T) {
		ai.TestProcessAttachments(t, model)
	})

	t.Run("GenerateContentWithTools", func(t *testing.T) {
		ai.TestGenerateContentWithTools(t, model)
	})
}
