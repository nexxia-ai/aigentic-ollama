package ollama

import (
	"os"
	"testing"

	"github.com/nexxia-ai/aigentic/ai"
)

// TestOpenAI_StandardSuite runs the standard test suite against the OpenAI implementation
func TestOpenAI_StandardSuite(t *testing.T) {
	suite := ai.ModelTestSuite{
		NewModel: func() *ai.Model {
			return NewOllamaModel("qwen3:1.7b", "")
		},
		Name: "Ollama",
	}
	ai.RunModelTestSuite(t, suite)
}

// TestOpenAI_IndividualTests demonstrates how to run individual tests
func TestOpenAI_IndividualTests(t *testing.T) {
	model := NewOllamaModel("qwen3:1.7b", "")

	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

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
