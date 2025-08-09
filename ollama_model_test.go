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
		SkipTests: []string{"ProcessImage"},
		Name:      "Ollama",
	}
	ai.RunModelTestSuite(t, suite)
}

func TestOllama_ProcessImage(t *testing.T) {
	model := NewModel("qwen3:4b", "")
	ai.TestProcessImage(t, model)
}
