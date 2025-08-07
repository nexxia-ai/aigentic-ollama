package ollama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/nexxia-ai/aigentic/ai"
)

// Ollama API types for native REST calls
type OllamaChatRequest struct {
	Model    string          `json:"model"`
	Messages []OllamaMessage `json:"messages"`
	Tools    []OllamaTool    `json:"tools,omitempty"`
	Options  *OllamaOptions  `json:"options,omitempty"`
	Stream   bool            `json:"stream,omitempty"`
}

type OllamaMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	Images    []string         `json:"images,omitempty"`
	ToolCalls []OllamaToolCall `json:"tool_calls,omitempty"`
}

type OllamaToolCall struct {
	ID       string         `json:"id"`
	Type     string         `json:"type"`
	Function OllamaFunction `json:"function"`
}

type OllamaFunction struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
	Index     int                    `json:"index"`
}

type OllamaTool struct {
	Type     string             `json:"type"`
	Function OllamaToolFunction `json:"function"`
}

type OllamaToolFunction struct {
	Name        string               `json:"name"`
	Description string               `json:"description"`
	Parameters  OllamaToolParameters `json:"parameters"`
}

type OllamaToolParameters struct {
	Type       string                                 `json:"type"`
	Properties map[string]OllamaToolParameterProperty `json:"properties"`
	Required   []string                               `json:"required,omitempty"`
}

type OllamaToolParameterProperty struct {
	Type        string `json:"type"`
	Description string `json:"description,omitempty"`
	Items       any    `json:"items,omitempty"`
	Enum        []any  `json:"enum,omitempty"`
}

type OllamaOptions struct {
	NumKeep          int      `json:"num_keep,omitempty"`
	Seed             int      `json:"seed,omitempty"`
	NumPredict       int      `json:"num_predict,omitempty"`
	TopK             int      `json:"top_k,omitempty"`
	TopP             float64  `json:"top_p,omitempty"`
	MinP             float64  `json:"min_p,omitempty"`
	TypicalP         float64  `json:"typical_p,omitempty"`
	RepeatLastN      int      `json:"repeat_last_n,omitempty"`
	Temperature      float64  `json:"temperature,omitempty"`
	RepeatPenalty    float64  `json:"repeat_penalty,omitempty"`
	PresencePenalty  float64  `json:"presence_penalty,omitempty"`
	FrequencyPenalty float64  `json:"frequency_penalty,omitempty"`
	PenalizeNewline  bool     `json:"penalize_newline,omitempty"`
	Stop             []string `json:"stop,omitempty"`
	Numa             bool     `json:"numa,omitempty"`
	NumCtx           int      `json:"num_ctx,omitempty"`
	NumBatch         int      `json:"num_batch,omitempty"`
	NumGpu           int      `json:"num_gpu,omitempty"`
	MainGpu          int      `json:"main_gpu,omitempty"`
	UseMmap          bool     `json:"use_mmap,omitempty"`
	NumThread        int      `json:"num_thread,omitempty"`
}

type OllamaChatResponse struct {
	Model              string        `json:"model"`
	CreatedAt          string        `json:"created_at"`
	Message            OllamaMessage `json:"message"`
	Done               bool          `json:"done"`
	TotalDuration      int64         `json:"total_duration,omitempty"`
	LoadDuration       int64         `json:"load_duration,omitempty"`
	PromptEvalCount    int           `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64         `json:"prompt_eval_duration,omitempty"`
	EvalCount          int           `json:"eval_count,omitempty"`
	EvalDuration       int64         `json:"eval_duration,omitempty"`
}

type StatusError struct {
	StatusCode   int
	Status       string
	ErrorMessage string
}

func (e StatusError) Error() string {
	return fmt.Sprintf("status: %s, code: %d, error: %s", e.Status, e.StatusCode, e.ErrorMessage)
}

// isRetryableError checks if an error should trigger a retry
func isRetryableError(err error) error {
	if err == nil {
		return nil
	}

	errStr := err.Error()

	// Check for specific HTTP status codes that are retryable
	if strings.Contains(errStr, "status: 502") ||
		strings.Contains(errStr, "status: 503") ||
		strings.Contains(errStr, "status: 504") ||
		strings.Contains(errStr, "status: 429") {
		return fmt.Errorf("%w: %v", ai.ErrTemporary, err)
	}

	// Check for network-related errors
	if strings.Contains(errStr, "connection refused") ||
		strings.Contains(errStr, "timeout") ||
		strings.Contains(errStr, "network") ||
		strings.Contains(errStr, "temporary") {
		return fmt.Errorf("%w: %v", ai.ErrTemporary, err)
	}

	return err
}

// NewModel creates a new Model instance configured for Ollama
func NewModel(modelName string, apiKey string) *ai.Model {
	if apiKey == "" {
		apiKey = os.Getenv("OLLAMA_API_KEY")
	}

	model := &ai.Model{
		ModelName: modelName,
		APIKey:    apiKey,
		BaseURL:   "http://localhost:11434",
	}
	model.SetGenerateFunc(ollamaGenerate)
	model.SetStreamingFunc(ollamaStream)

	return model
}

// ollamaGenerate is the generate function for Ollama models
func ollamaGenerate(ctx context.Context, model *ai.Model, messages []ai.Message, tools []ai.Tool) (ai.AIMessage, error) {
	// Process messages with aggregation for Ollama's constraints
	var systemMessage *OllamaMessage
	var userMessage *OllamaMessage
	var imageMessages []OllamaMessage
	var conversationMessages []OllamaMessage

	for _, msg := range messages {
		switch r := msg.(type) {
		case ai.SystemMessage:
			// Aggregate system messages
			if systemMessage == nil {
				systemMessage = &OllamaMessage{
					Role:    "system",
					Content: r.Content,
				}
			} else {
				systemMessage.Content += "\n" + r.Content
			}

		case ai.UserMessage:
			// Aggregate user messages
			if userMessage == nil {
				userMessage = &OllamaMessage{
					Role:    "user",
					Content: r.Content,
				}
			} else {
				userMessage.Content += "\n" + r.Content
			}

		case ai.ResourceMessage:
			// Handle based on resource type
			switch r.Type {
			case "image":
				// Images become separate messages
				if r.MIMEType != "" && r.Body != nil {
					imageData := base64.StdEncoding.EncodeToString(r.Body.([]byte))
					imageMessages = append(imageMessages, OllamaMessage{
						Role:   "user",
						Images: []string{imageData},
					})
				}

			default:
				// All other resource types embedded in user message with demarcation
				if userMessage == nil {
					userMessage = &OllamaMessage{Role: "user"}
				}

				// Build file demarcation
				fileMarker := "<file"
				if r.Name != "" {
					fileMarker += fmt.Sprintf(` name="%s"`, r.Name)
				}
				if r.MIMEType != "" {
					fileMarker += fmt.Sprintf(` type="%s"`, r.MIMEType)
				}
				fileMarker += ">"

				// Add file content
				var fileContent string
				if r.Body != nil {
					switch r.MIMEType {
					case "text/plain", "text/markdown", "text/csv", "application/json", "text/html":
						// Embed text content directly
						if textContent, ok := r.Body.([]byte); ok {
							fileContent = string(textContent)
						}
					default:
						// For binary files, add a reference
						fileContent = fmt.Sprintf("[Binary file: %s]", r.MIMEType)
					}
				}

				// Add description if available
				if r.Description != "" {
					fileContent = r.Description + "\n\n" + fileContent
				}

				// Add to user message with demarcation
				userMessage.Content += fileMarker + fileContent + "</file>\n\n"
			}

		case ai.AIMessage:
			// Handle AI messages normally
			ollamaMsg := OllamaMessage{
				Role:    "assistant",
				Content: r.OriginalContent,
			}

			// Convert tool calls if any
			for _, toolCall := range r.ToolCalls {
				ollamaMsg.ToolCalls = append(ollamaMsg.ToolCalls, OllamaToolCall{
					ID:   toolCall.ID,
					Type: toolCall.Type,
					Function: OllamaFunction{
						Name:      toolCall.Name,
						Arguments: map[string]interface{}{},
						Index:     0, // Will be set properly if needed
					},
				})
			}

			conversationMessages = append(conversationMessages, ollamaMsg)

		case ai.ToolMessage:
			// Handle tool messages
			conversationMessages = append(conversationMessages, OllamaMessage{
				Role:    "tool",
				Content: r.Content,
			})

		default:
			panic(fmt.Sprintf("unsupported message type: %T - check that message is not a pointer", r))
		}
	}

	// Assemble final message list
	var ollamaMessages []OllamaMessage

	// Add system message if exists
	if systemMessage != nil {
		ollamaMessages = append(ollamaMessages, *systemMessage)
	}

	// Add user message if exists
	if userMessage != nil {
		ollamaMessages = append(ollamaMessages, *userMessage)
	}

	// Add all image messages
	ollamaMessages = append(ollamaMessages, imageMessages...)

	// Add remaining conversation messages
	ollamaMessages = append(ollamaMessages, conversationMessages...)

	// Convert our tool format to Ollama's format
	ollamaTools := make([]OllamaTool, len(tools))
	if len(tools) > 0 {
		for i, tool := range tools {
			ollamaTools[i] = OllamaTool{
				Type:     "function",
				Function: toToolFunction(tool),
			}
		}
	}

	// Make a single LLM call and return the result
	options := ollamaModelToOptions(model)
	apiRespMsg, err := ollamaREST(ctx, model, ollamaMessages, ollamaTools, options)
	if err != nil {
		return ai.AIMessage{}, fmt.Errorf("failed to call ollama generate: %w", err)
	}

	// Convert Ollama response to AIMessage
	content, thinkPart := ai.ExtractThinkTags(apiRespMsg.Content)
	finalMessage := ai.AIMessage{Role: ai.AssistantRole, Content: content, Think: thinkPart, OriginalContent: apiRespMsg.Content}

	// Convert tool calls if any
	for _, tc := range apiRespMsg.ToolCalls {
		args, err := json.Marshal(tc.Function.Arguments)
		if err != nil {
			return ai.AIMessage{}, fmt.Errorf("failed to marshal tool arguments: %w", err)
		}

		finalMessage.ToolCalls = append(finalMessage.ToolCalls, ai.ToolCall{
			ID:     strconv.Itoa(tc.Function.Index),
			Type:   tc.Type,
			Name:   tc.Function.Name,
			Args:   string(args),
			Result: "",
		})
	}

	return finalMessage, nil
}

// ollamaStream is the streaming function for Ollama models
func ollamaStream(ctx context.Context, model *ai.Model, messages []ai.Message, tools []ai.Tool, chunkFunction func(ai.AIMessage) error) (ai.AIMessage, error) {
	// Process messages with aggregation for Ollama's constraints
	var systemMessage *OllamaMessage
	var userMessage *OllamaMessage
	var imageMessages []OllamaMessage
	var conversationMessages []OllamaMessage

	for _, msg := range messages {
		switch r := msg.(type) {
		case ai.SystemMessage:
			// Aggregate system messages
			if systemMessage == nil {
				systemMessage = &OllamaMessage{
					Role:    "system",
					Content: r.Content,
				}
			} else {
				systemMessage.Content += "\n" + r.Content
			}

		case ai.UserMessage:
			// Aggregate user messages
			if userMessage == nil {
				userMessage = &OllamaMessage{
					Role:    "user",
					Content: r.Content,
				}
			} else {
				userMessage.Content += "\n" + r.Content
			}

		case ai.ResourceMessage:
			// Handle based on resource type
			switch r.Type {
			case "image":
				// Images become separate messages
				if r.MIMEType != "" && r.Body != nil {
					imageData := base64.StdEncoding.EncodeToString(r.Body.([]byte))
					imageMessages = append(imageMessages, OllamaMessage{
						Role:   "user",
						Images: []string{imageData},
					})
				}

			default:
				// All other resource types embedded in user message with demarcation
				if userMessage == nil {
					userMessage = &OllamaMessage{Role: "user"}
				}

				// Build file demarcation
				fileMarker := "<file"
				if r.Name != "" {
					fileMarker += fmt.Sprintf(` name="%s"`, r.Name)
				}
				if r.MIMEType != "" {
					fileMarker += fmt.Sprintf(` type="%s"`, r.MIMEType)
				}
				fileMarker += ">"

				// Add file content
				var fileContent string
				if r.Body != nil {
					switch r.MIMEType {
					case "text/plain", "text/markdown", "text/csv", "application/json", "text/html":
						// Embed text content directly
						if textContent, ok := r.Body.([]byte); ok {
							fileContent = string(textContent)
						}
					default:
						// For binary files, add a reference
						fileContent = fmt.Sprintf("[Binary file: %s]", r.MIMEType)
					}
				}

				// Add description if available
				if r.Description != "" {
					fileContent = r.Description + "\n\n" + fileContent
				}

				// Add to user message with demarcation
				userMessage.Content += fileMarker + fileContent + "</file>\n\n"
			}

		case ai.AIMessage:
			// Handle AI messages normally
			ollamaMsg := OllamaMessage{
				Role:    "assistant",
				Content: r.OriginalContent,
			}

			// Convert tool calls if any
			for _, toolCall := range r.ToolCalls {
				ollamaMsg.ToolCalls = append(ollamaMsg.ToolCalls, OllamaToolCall{
					ID:   toolCall.ID,
					Type: toolCall.Type,
					Function: OllamaFunction{
						Name:      toolCall.Name,
						Arguments: map[string]interface{}{},
						Index:     0, // Will be set properly if needed
					},
				})
			}

			conversationMessages = append(conversationMessages, ollamaMsg)

		case ai.ToolMessage:
			// Handle tool messages
			conversationMessages = append(conversationMessages, OllamaMessage{
				Role:    "tool",
				Content: r.Content,
			})

		default:
			panic(fmt.Sprintf("unsupported message type: %T - check that message is not a pointer", r))
		}
	}

	// Assemble final message list
	var ollamaMessages []OllamaMessage

	// Add system message if exists
	if systemMessage != nil {
		ollamaMessages = append(ollamaMessages, *systemMessage)
	}

	// Add user message if exists
	if userMessage != nil {
		ollamaMessages = append(ollamaMessages, *userMessage)
	}

	// Add all image messages
	ollamaMessages = append(ollamaMessages, imageMessages...)

	// Add remaining conversation messages
	ollamaMessages = append(ollamaMessages, conversationMessages...)

	// Convert our tool format to Ollama's format
	ollamaTools := make([]OllamaTool, len(tools))
	if len(tools) > 0 {
		for i, tool := range tools {
			ollamaTools[i] = OllamaTool{
				Type:     "function",
				Function: toToolFunction(tool),
			}
		}
	}

	// Make streaming call to Ollama
	options := ollamaModelToOptions(model)
	finalMessage, err := ollamaStreamREST(ctx, model, ollamaMessages, ollamaTools, options, chunkFunction)
	if err != nil {
		return ai.AIMessage{}, fmt.Errorf("failed to call ollama stream: %w", err)
	}

	return finalMessage, nil
}

// ollamaStreamREST makes a streaming call to the Ollama API
func ollamaStreamREST(ctx context.Context, model *ai.Model, messages []OllamaMessage, tools []OllamaTool, options *OllamaOptions, chunkFunction func(ai.AIMessage) error) (ai.AIMessage, error) {
	req := &OllamaChatRequest{
		Model:    model.ModelName,
		Messages: messages,
		Stream:   true, // Enable streaming
	}

	// Add tools to the request if provided
	if len(tools) > 0 {
		req.Tools = tools
	}

	// Set options if provided
	if options != nil {
		req.Options = options
	}

	// Marshal request to JSON
	reqBody, err := json.Marshal(req)
	if err != nil {
		return ai.AIMessage{}, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	url := model.BaseURL + "/api/chat"
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return ai.AIMessage{}, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if model.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+model.APIKey)
	}

	// Execute request
	client := &http.Client{}
	resp, err := client.Do(httpReq)
	if err != nil {
		return ai.AIMessage{}, isRetryableError(err)
	}
	defer resp.Body.Close()

	// Check for HTTP errors first
	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		errStatus := &StatusError{
			StatusCode:   resp.StatusCode,
			Status:       resp.Status,
			ErrorMessage: string(respBody),
		}
		return ai.AIMessage{}, isRetryableError(errStatus)
	}

	// Handle streaming response
	var accumulatedContent string
	var accumulatedThinkContent string
	var accumulatedToolCalls []OllamaToolCall
	var finalMessage ai.AIMessage
	scanner := bufio.NewScanner(resp.Body)

	for scanner.Scan() {
		// Check for context cancellation
		select {
		case <-ctx.Done():
			return ai.AIMessage{}, ctx.Err()
		default:
		}

		line := scanner.Text()
		if line == "" {
			continue
		}

		var chunk OllamaChatResponse
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			return ai.AIMessage{}, fmt.Errorf("failed to unmarshal response chunk: %w", err)
		}

		// Initialize final message with first chunk
		if finalMessage.Role == "" {
			finalMessage.Role = ai.AssistantRole
		}

		// Extract think tags from this chunk's content
		chunkContent, chunkThinkPart := ai.ExtractThinkTags(chunk.Message.Content)

		// Accumulate cleaned content (without think tags)
		if chunkContent != "" {
			accumulatedContent += chunkContent
		}

		// Accumulate think content
		if chunkThinkPart != "" {
			accumulatedThinkContent += chunkThinkPart
		}

		// Accumulate tool calls
		if chunk.Message.ToolCalls != nil {
			accumulatedToolCalls = append(accumulatedToolCalls, chunk.Message.ToolCalls...)
		}

		// Create chunk message for callback with only the new cleaned content (no think tags)
		chunkMessage := ai.AIMessage{
			Role:            ai.AssistantRole,
			Content:         chunkContent, // Only the new cleaned content for this chunk
			Think:           chunkThinkPart,
			OriginalContent: chunk.Message.Content, // Keep original for reference
		}

		// Convert only the new tool calls for this chunk
		if chunk.Message.ToolCalls != nil {
			for _, tc := range chunk.Message.ToolCalls {
				args, err := json.Marshal(tc.Function.Arguments)
				if err != nil {
					return ai.AIMessage{}, fmt.Errorf("failed to marshal tool arguments: %w", err)
				}

				chunkMessage.ToolCalls = append(chunkMessage.ToolCalls, ai.ToolCall{
					ID:     strconv.Itoa(tc.Function.Index),
					Type:   tc.Type,
					Name:   tc.Function.Name,
					Args:   string(args),
					Result: "",
				})
			}
		}

		// Call the chunk function
		if err := chunkFunction(chunkMessage); err != nil {
			return ai.AIMessage{}, fmt.Errorf("chunk function error: %w", err)
		}

		// Check if done
		if chunk.Done {
			break
		}
	}

	if err := scanner.Err(); err != nil {
		return ai.AIMessage{}, fmt.Errorf("error reading response: %w", err)
	}

	// Set final message with accumulated cleaned content and think content
	finalMessage.Content = accumulatedContent
	finalMessage.Think = accumulatedThinkContent
	finalMessage.OriginalContent = accumulatedContent + accumulatedThinkContent

	// Convert final tool calls
	for _, tc := range accumulatedToolCalls {
		args, err := json.Marshal(tc.Function.Arguments)
		if err != nil {
			return ai.AIMessage{}, fmt.Errorf("failed to marshal tool arguments: %w", err)
		}

		finalMessage.ToolCalls = append(finalMessage.ToolCalls, ai.ToolCall{
			ID:     strconv.Itoa(tc.Function.Index),
			Type:   tc.Type,
			Name:   tc.Function.Name,
			Args:   string(args),
			Result: "",
		})
	}

	return finalMessage, nil
}

// ollamaModelToOptions converts model pointer fields to OllamaOptions
func ollamaModelToOptions(model *ai.Model) *OllamaOptions {
	// Check if any options are set on the model
	if model.Temperature == nil && model.MaxTokens == nil && model.TopP == nil &&
		model.FrequencyPenalty == nil && model.PresencePenalty == nil &&
		model.StopSequences == nil && model.ContextSize == nil {
		return nil
	}

	options := &OllamaOptions{}

	// Only set values that were explicitly set (non-nil pointers)
	if model.ContextSize != nil {
		options.NumCtx = *model.ContextSize
	}
	if model.Temperature != nil {
		options.Temperature = *model.Temperature
	}
	if model.TopP != nil {
		options.TopP = *model.TopP
	}
	if model.MaxTokens != nil {
		options.NumPredict = *model.MaxTokens
	}
	if model.FrequencyPenalty != nil {
		options.FrequencyPenalty = *model.FrequencyPenalty
	}
	if model.PresencePenalty != nil {
		options.PresencePenalty = *model.PresencePenalty
	}
	if model.StopSequences != nil {
		options.Stop = *model.StopSequences
	}

	return options
}

// ollamaREST makes a single call to the Ollama API
func ollamaREST(ctx context.Context, model *ai.Model, messages []OllamaMessage, tools []OllamaTool, options *OllamaOptions) (OllamaMessage, error) {
	req := &OllamaChatRequest{
		Model:    model.ModelName,
		Messages: messages,
	}

	// Add tools to the request if provided
	if len(tools) > 0 {
		req.Tools = tools
	}

	// Set options if provided
	if options != nil {
		req.Options = options
	}

	// Marshal request to JSON
	reqBody, err := json.Marshal(req)
	if err != nil {
		return OllamaMessage{}, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	url := model.BaseURL + "/api/chat"
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return OllamaMessage{}, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if model.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+model.APIKey)
	}

	// Execute request
	client := &http.Client{}
	resp, err := client.Do(httpReq)
	if err != nil {
		return OllamaMessage{}, isRetryableError(err)
	}
	defer resp.Body.Close()

	// Check for HTTP errors first
	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		errStatus := &StatusError{
			StatusCode:   resp.StatusCode,
			Status:       resp.Status,
			ErrorMessage: string(respBody),
		}
		return OllamaMessage{}, isRetryableError(errStatus)
	}

	// Handle streaming response
	var responseMessage OllamaMessage
	scanner := bufio.NewScanner(resp.Body)

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		var chunk OllamaChatResponse
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			return OllamaMessage{}, fmt.Errorf("failed to unmarshal response chunk: %w", err)
		}

		// Initialize response message with first chunk
		if responseMessage.Role == "" {
			responseMessage.Role = chunk.Message.Role
		}

		// Accumulate content
		responseMessage.Content += chunk.Message.Content

		// Accumulate tool calls
		if chunk.Message.ToolCalls != nil {
			responseMessage.ToolCalls = append(responseMessage.ToolCalls, chunk.Message.ToolCalls...)
		}

		// Check if done
		if chunk.Done {
			break
		}
	}

	if err := scanner.Err(); err != nil {
		return OllamaMessage{}, fmt.Errorf("error reading response: %w", err)
	}

	return responseMessage, nil
}

// toToolFunction converts a SimpleTool to OllamaToolFunction
func toToolFunction(tool ai.Tool) OllamaToolFunction {
	// Ollama api uses anonymous structs for properties making it difficult to map to our tool input schema
	// We need to convert our tool input schema to Ollama's format
	apiProperties := make(map[string]OllamaToolParameterProperty)

	toolFunction := OllamaToolFunction{}
	toolFunction.Name = tool.Name
	toolFunction.Description = tool.Description
	toolFunction.Parameters.Type = "object"

	// Convert the input schema to Ollama format
	if tool.InputSchema != nil {
		if required, ok := tool.InputSchema["required"].([]string); ok {
			toolFunction.Parameters.Required = required
		}

		if properties, ok := tool.InputSchema["properties"].(map[string]interface{}); ok {
			for propName, propValue := range properties {
				if propMap, ok := propValue.(map[string]interface{}); ok {
					propStruct := OllamaToolParameterProperty{}

					if propType, ok := propMap["type"].(string); ok {
						propStruct.Type = propType
					}
					if propDesc, ok := propMap["description"].(string); ok {
						propStruct.Description = propDesc
					}
					if propEnum, ok := propMap["enum"].([]interface{}); ok {
						propStruct.Enum = propEnum
					}

					apiProperties[propName] = propStruct
				}
			}
		}
	}

	toolFunction.Parameters.Properties = apiProperties

	return toolFunction
}
