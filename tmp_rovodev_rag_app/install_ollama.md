# Ollama Installation Instructions

## Option 1: Download and Install Ollama

### Windows:
1. **Download Ollama**: Go to https://ollama.ai/download/windows
2. **Install**: Run the downloaded installer
3. **Restart** your terminal/PowerShell
4. **Install a model**: 
   ```bash
   ollama pull llama2
   ```
   Or for a smaller, faster model:
   ```bash
   ollama pull phi
   ```

### Alternative Models:
- **llama2** (7GB) - Good quality, general purpose
- **phi** (1.6GB) - Smaller, faster, Microsoft model
- **mistral** (4GB) - Good balance of speed and quality
- **codellama** (7GB) - Better for code-related questions

## Option 2: Use Alternative Local AI

If you prefer not to install Ollama, you can:

1. **Use OpenAI** (requires API key and billing)
2. **Use Text-Only Mode** (see retrieved documents without AI answers)
3. **Use Hugging Face models** (requires more setup)

## After Installing Ollama:

1. **Test connection**:
   ```bash
   cd tmp_rovodev_rag_app
   python test_ollama.py
   ```

2. **Test RAG system**:
   ```bash
   python run_app.py query "What are operating systems?"
   ```

3. **Start web interface**:
   ```bash
   python run_app.py web
   ```

## Why Ollama?

- ✅ **Private**: Runs locally, no data sent to external servers
- ✅ **Free**: No API costs after initial setup
- ✅ **Fast**: Direct hardware acceleration
- ✅ **Flexible**: Many models available