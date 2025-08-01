//go:build integration

// run this with: go test -v -tags=integration -run ^TestOllama_AgentSuite

package ollama

import (
	"testing"

	"github.com/nexxia-ai/aigentic"
	"github.com/nexxia-ai/aigentic/ai"
)

func TestOllama_AgentSuite(t *testing.T) {
	aigentic.RunIntegrationTestSuite(t, aigentic.IntegrationTestSuite{
		NewModel: func() *ai.Model {
			return NewModel("qwen3:4b", "")
		},
		Name:      "Ollama",
		SkipTests: []string{}, // Ollama supports all test types
	})
}
