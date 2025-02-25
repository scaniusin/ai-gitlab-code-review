# GitLab AI Code Reviewer with Ollama Integration

This service provides automated code review for GitLab merge requests using the Qwen 2.5 Coder (32B) model through Ollama. It analyzes code changes and provides detailed feedback on code quality, potential issues, and suggested improvements.

## Features

- Automated code review using Qwen 2.5 Coder (32B)
- Integration with GitLab merge requests
- Local model execution through Ollama
- Detailed code analysis and suggestions
- Markdown-formatted responses for GitLab

## Prerequisites

- Docker and Docker Compose
- At least 32GB RAM (recommended for running the Qwen model)
- Approximately 60GB free disk space for the model

## Quick Start Guide

1. Clone the repository and navigate to the gitlab-review directory:
```bash
cd docker/gitlab-review
```

2. Create your environment file:
```bash
cp .env.sample .env
```

3. Edit the `.env` file with your configuration:
```env
# GitLab Configuration
GITLAB_TOKEN=<your GitLab API token>
GITLAB_URL=https://gitlab.com/api/v4
EXPECTED_GITLAB_TOKEN=<your webhook token>

# AI Service Configuration
# Choose which AI service to use: 'openai' or 'ollama'
AI_SERVICE_TYPE=openai

# OpenAI Configuration (required if AI_SERVICE_TYPE=openai)
OPENAI_API_KEY=<your OpenAI API key>
OPENAI_API_MODEL=gpt-3.5-turbo

# Azure OpenAI Configuration (optional)
#AZURE_OPENAI_API_BASE=<your Azure OpenAI endpoint>
#AZURE_OPENAI_API_VERSION=<API version>

# Ollama Configuration (required if AI_SERVICE_TYPE=ollama)
#OLLAMA_HOST=http://ollama:11434
#OLLAMA_MODEL=custom-qwen
```

4. Start the services:
```bash
docker compose up -d
```

5. Pull the Qwen model (only needed once, as it will be saved to the volume):
```bash
docker exec fn_ollama ollama pull qwen2.5-coder:32b
```

6. Create the custom model with our configuration:
```bash
docker exec -w /root fn_ollama ollama create custom-qwen -f Modelfile.qwen
```

7. Test the model:
```bash
docker exec fn_ollama ollama run custom-qwen "Write a simple hello world in Python"
```

## Usage

1. Configure your GitLab repository webhooks:
   - Go to your repository's Settings > Webhooks
   - Add a new webhook with URL: `http://your-server:80/webhook`
   - Set the Secret Token from your .env file
   - Select events: Merge request events, Push events

2. The service will automatically:
   - Review new merge requests
   - Analyze code changes
   - Post comments with suggestions and improvements

## Troubleshooting

1. If the Ollama service fails to start:
   - Check system resources (RAM, disk space)
   - Review logs: `docker compose logs ollama`

2. If the GitLab review service fails:
   - Verify environment variables
   - Check logs: `docker compose logs gitlab-review`

3. If the model is slow:
   - Ensure sufficient system resources
   - Consider reducing concurrent requests

## Resource Requirements

- RAM: 32GB minimum (48GB recommended)
- Disk Space: 60GB for model + 10GB for container images
- CPU: 4+ cores recommended

For production use, consider adjusting these requirements based on your workload.
