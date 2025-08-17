//go:build integration

// run this with: go test -v -tags=integration -run ^TestOllama_AgentSuite

package ollama

import (
	"testing"

	"github.com/nexxia-ai/aigentic"
	"github.com/nexxia-ai/aigentic/ai"
)

const (
	gptModel = "gpt-oss:20b"
)

func TestOllama_AgentSuite(t *testing.T) {
	aigentic.RunIntegrationTestSuite(t, aigentic.IntegrationTestSuite{
		NewModel: func() *ai.Model {
			return NewModel("qwen3:8b", "")
		},
		Name: "Ollama",
		SkipTests: []string{
			"MultiAgentChain",
			"TeamCoordination",
			"MemoryPersistence",
			"FileAttachments"}, // qwen3:4b is not strong enough for this test
	})
}

func TestOllama_MemoryPersistence(t *testing.T) {
	model := NewModel("qwen3:8b", "")
	aigentic.TestMemoryPersistence(t, model)
}

func TestOllama_MultiAgentChain(t *testing.T) {
	// qwen3:14b is still not strong enough for this test either
	model := NewModel("qwen3:14b", "")
	aigentic.TestMultiAgentChain(t, model)
}

func TestOllama_TeamCoordination(t *testing.T) {
	model := NewModel("qwen3:14b", "")
	aigentic.TestTeamCoordination(t, model)
}

func TestOllama_FileAttachments(t *testing.T) {
	model := NewModel("qwen3:8b", "")
	aigentic.TestFileAttachments(t, model)
}
