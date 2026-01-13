# aigentic-ollama

Import this module to register local Ollama models with the `ai` registry. Models are created with `ai.New` using the registered identifier:

```go
import (
	"github.com/nexxia-ai/aigentic/ai"
	_ "github.com/nexxia-ai/aigentic-ollama"
)

model, err := ai.New("Qwen3 4B", "")
if err != nil {
	// handle error
}
```