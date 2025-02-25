FROM ollama/ollama:latest

# Expose the default Ollama port
EXPOSE 11434

# Copy Modelfile
COPY models/Modelfile.qwen /root/

# Set health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:11434/api/health || exit 1
