# LLM Backend Setup Guide

Your crisis assistant now supports multiple LLM backends! Here's how to set them up:

## Current Status ✅
- **Mock LLM**: Currently active and working perfectly for testing
- **Advanced RAG**: Working with hybrid search, reranking, and 249 medical scenarios
- **Smart Retrieval**: Finding relevant medical information with high accuracy

## Backend Options

### 1. Mock LLM (Currently Active)
**Status**: ✅ Working  
**Use case**: Testing, development, offline demos  
**Configuration**: Already set up in `configs/config.yaml`

```yaml
llm:
  backend: "mock"
```

### 2. Ollama (Recommended for Local Use)
**Status**: 🔧 Available  
**Use case**: Local, private, no internet required  
**Setup**:

1. Install Ollama: https://ollama.ai/
2. Pull a model: `ollama pull llama3.2:3b`
3. Update config:
```yaml
llm:
  backend: "ollama"
  ollama_config:
    model: "llama3.2:3b"  # or phi3, mistral, etc.
```

### 3. OpenAI GPT (Cloud)
**Status**: 🔧 Available  
**Use case**: High quality responses, reliable  
**Setup**:

1. Install: `pip install openai`
2. Get API key from https://platform.openai.com/
3. Set environment variable: `export OPENAI_API_KEY="your-key"`
4. Update config:
```yaml
llm:
  backend: "openai"
  openai_config:
    model: "gpt-3.5-turbo"
```

### 4. Anthropic Claude (Cloud)
**Status**: 🔧 Available  
**Use case**: High quality, safety-focused  
**Setup**:

1. Install: `pip install anthropic`
2. Get API key from https://console.anthropic.com/
3. Set environment variable: `export ANTHROPIC_API_KEY="your-key"`
4. Update config:
```yaml
llm:
  backend: "anthropic"
  anthropic_config:
    model: "claude-3-haiku-20240307"
```

## Quick Switch Instructions

To switch backends, just edit `configs/config.yaml`:

```yaml
llm:
  backend: "ollama"  # Change this line
```

The system will automatically fall back to mock if the backend fails.

## Performance Comparison

| Backend | Speed | Quality | Privacy | Cost |
|---------|-------|---------|---------|------|
| Mock | ⚡⚡⚡ | ⭐⭐ | 🔒🔒🔒 | 💰 |
| Ollama | ⚡⚡ | ⭐⭐⭐ | 🔒🔒🔒 | 💰 |
| OpenAI | ⚡⚡⚡ | ⭐⭐⭐⭐ | 🔒 | 💰💰💰 |
| Claude | ⚡⚡ | ⭐⭐⭐⭐ | 🔒 | 💰💰💰💰 |

## Current Features Working ✅

- **Advanced RAG System**: Hybrid search with dense + sparse retrieval
- **Medical Database**: 249 comprehensive emergency scenarios
- **Smart Reranking**: Cross-encoder reranking for best results  
- **Context-Aware**: Emergency type detection and severity assessment
- **Sources**: Shows authoritative medical sources for each response

## Next: Enable Advanced Features

Once you choose your preferred backend, we can:
- Re-enable enhanced prompt engineering
- Add response validation
- Enable performance monitoring
- Add safety guardrails