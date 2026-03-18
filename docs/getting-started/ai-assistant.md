# AI workflow assistant (beta)

The AI assistant generates node graphs from text descriptions. Describe what you want to do and it builds the pipeline for you.

Open it from **View > AI Assistant** in the menu bar.

## Setup

1. Select a **Provider** from the dropdown
2. Paste your **API Key** (not needed for local providers such as Ollama)
3. Click the refresh button to load available models
4. Pick a model

### Providers

| Provider | API Key | Notes |
|----------|---------|-------|
| Ollama | not needed | Local. Install from [ollama.com](https://ollama.com), then pull a model (`ollama pull gemma3:12b`) |
| Ollama Cloud | needed | Create an account at [ollama.com](https://ollama.com), then get a key at [ollama.com/settings/keys](https://ollama.com/settings/keys) |
| OpenAI | needed | Get a key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| Claude | needed | Get a key at [console.anthropic.com](https://console.anthropic.com) |
| Groq | needed | Free tier available at [console.groq.com](https://console.groq.com) |
| Gemini | needed | Free tier available at [aistudio.google.com](https://aistudio.google.com) |
| RunPod | needed | Serverless vLLM endpoint. Enter your Endpoint ID separately |

API keys are stored locally and never sent anywhere except the selected provider.

## Usage

1. Type a description of the workflow you want, e.g.:

    > Load a CSV, filter out data with area > 100, then make a bar plot

2. **Include current workflow as context** (optional): check this to let the AI see and modify your existing pipeline instead of building from scratch

3. **Verbose node descriptions** (optional): sends full docstrings to the model for better context. Larger prompt but more accurate results.

4. Click **Generate Workflow**. This takes 10-60 seconds depending on the model.

5. Review the generated JSON in the preview panel

6. Click **Load into Canvas** to append the nodes, or **Replace Canvas** to start fresh

## Tips

- Simpler prompts work better. Break complex pipelines into steps.
- If the model gets a port name wrong (you'll see warnings), try regenerating or fix the connection manually.