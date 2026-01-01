//go:build integration

// run this with: go test -v -tags=integration -run ^TestOllama_AgentSuite

package ollama

import (
	"testing"

	"github.com/nexxia-ai/aigentic"
	"github.com/nexxia-ai/aigentic/ai"
	"github.com/nexxia-ai/aigentic/event"
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

func TestOllama_ThinkingEvents(t *testing.T) {
	model := NewModel("qwen3:8b", "")
	maxTokens := 500
	model.MaxTokens = &maxTokens

	agent := aigentic.Agent{
		Name:        "ThinkingAgent",
		Description: "Test agent for thinking events",
		Instructions: `You are a helpful assistant. When answering questions, use <think> tags to show your reasoning process.
For example: <think>First, I should consider...</think> Then provide your answer.
Always use thinking tags when solving problems.`,
		Model:  model,
		Stream: true,
	}

	agentRun, err := agent.Start("What is 25 * 37? Show your thinking process in <think> tags.")
	if err != nil {
		t.Fatalf("Failed to start agent: %v", err)
	}

	var thinkingEvents []*event.ThinkingEvent
	var contentEvents []*event.ContentEvent

	for evt := range agentRun.Next() {
		switch ev := evt.(type) {
		case *event.ThinkingEvent:
			t.Logf("ThinkingEvent received: %s", ev.Thought)
			thinkingEvents = append(thinkingEvents, ev)
		case *event.ContentEvent:
			t.Logf("ContentEvent received: %s", ev.Content)
			contentEvents = append(contentEvents, ev)
		case *event.ErrorEvent:
			t.Fatalf("ErrorEvent received: %v", ev.Err)
		}
	}

	if len(thinkingEvents) == 0 {
		t.Fatal("Expected at least one ThinkingEvent but got none - BUG CONFIRMED")
	}

	fullThought := ""
	for _, te := range thinkingEvents {
		fullThought += te.Thought
	}

	if fullThought == "" {
		t.Fatal("ThinkingEvents were empty - BUG CONFIRMED")
	}

	t.Logf("Total thinking events: %d", len(thinkingEvents))
	t.Logf("Full thought content: %s", fullThought)
	t.Logf("Total content events: %d", len(contentEvents))
}
