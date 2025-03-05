import os
import json
import requests
import logging
from abc import ABC, abstractmethod
from flask import Flask, request
import openai
from typing import List, Dict, Any, Optional

# Константы для русских комментариев в ревью
REVIEW_COMMENTS = {
    'SUMMARY': 'Краткое описание изменений:',
    'CODE_CLARITY': 'Ясность кода:',
    'NAMING': 'Именование и комментарии:',
    'COMPLEXITY': 'Сложность кода:',
    'BUGS': 'Потенциальные ошибки:',
    'SECURITY': 'Проблемы безопасности:',
    'BEST_PRACTICES': 'Рекомендации по улучшению:',
    'LINE_COMMENT': 'Строка {}: {}'
}

# Промпты для анализа кода
MR_PRE_PROMPT = """Проанализируйте следующие изменения кода построчно. Для каждой измененной строки:
1. Укажите потенциальные проблемы
2. Предложите улучшения
3. Отметьте хорошие практики

В конце предоставьте краткое общее резюме (не более 3 предложений).
Используйте русский язык для всех комментариев."""

MR_LINE_ANALYSIS_PROMPT = """Для каждой измененной строки кода предоставьте:
1. Номер строки
2. Анализ изменения
3. Конкретные предложения по улучшению
4. Потенциальные проблемы безопасности или производительности"""

COMMIT_PRE_PROMPT = "Review the git diff of a recent commit, focusing on clarity, structure, and security."

COMMIT_QUESTIONS = """
Questions:
1. Summarize changes (Changelog style).
2. Clarity of added/modified code?
3. Comments and naming adequacy?
4. Simplification without breaking functionality? Examples?
5. Any bugs? Where?
6. Potential security issues?
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class AIService(ABC):
    @abstractmethod
    def generate_review(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        pass

class OpenAIService(AIService):
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
            
        openai.api_key = self.api_key
        
        api_base = os.environ.get("AZURE_OPENAI_API_BASE")
        if api_base:
            openai.api_base = api_base
            
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
        if api_version:
            openai.api_type = "azure"
            openai.api_version = api_version
    
    def generate_review(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        try:
            # Check if we're using Azure OpenAI
            if os.environ.get("AZURE_OPENAI_API_BASE"):
                completions = openai.ChatCompletion.create(
                    model=os.environ.get("OPENAI_API_MODEL") or "gpt-3.5-turbo",
                    temperature=temperature,
                    stream=False,
                    messages=messages
                )
            else:
                # For newer OpenAI client versions
                try:
                    client = openai.OpenAI(api_key=self.api_key)
                    completions = client.chat.completions.create(
                        model=os.environ.get("OPENAI_API_MODEL") or "gpt-3.5-turbo",
                        temperature=temperature,
                        messages=messages
                    )
                    return completions.choices[0].message.content.strip()
                except (AttributeError, ImportError):
                    # Fall back to legacy version
                    completions = openai.ChatCompletion.create(
                        model=os.environ.get("OPENAI_API_MODEL") or "gpt-3.5-turbo",
                        temperature=temperature,
                        stream=False,
                        messages=messages
                    )
            return completions.choices[0].message["content"].strip()
        except Exception as e:
            logging.error(f"OpenAI API error: {str(e)}")
            raise

class OllamaService(AIService):
    def __init__(self):
        self.api_url = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
        self.model = os.environ.get("OLLAMA_MODEL", "custom-qwen")
        self.timeout = int(os.environ.get("OLLAMA_TIMEOUT", "120"))
        self.max_tokens = int(os.environ.get("OLLAMA_MAX_TOKENS", "1800"))  # Leave some room for response
    
    def _chunk_messages(self, messages: List[Dict[str, str]], max_tokens: int) -> List[List[Dict[str, str]]]:
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Always include system message in each chunk if present
        system_message = None
        if messages and messages[0].get("role") == "system":
            system_message = messages[0]
            messages = messages[1:]
        
        for msg in messages:
            msg_length = len(msg["content"].split())
            
            if current_length + msg_length > max_tokens and current_chunk:
                if system_message:
                    current_chunk.insert(0, system_message)
                chunks.append(current_chunk)
                current_chunk = []
                current_length = 0
            
            current_chunk.append(msg)
            current_length += msg_length
        
        if current_chunk:
            if system_message:
                current_chunk.insert(0, system_message)
            chunks.append(current_chunk)
        
        return chunks
    
    def generate_review(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        try:
            # Split messages into chunks if they're too long
            message_chunks = self._chunk_messages(messages, self.max_tokens)
            responses = []
            
            for chunk in message_chunks:
                # Convert OpenAI-style messages to Ollama format
                prompt = "\n\n".join([msg["content"] for msg in chunk])
                
                logging.info(f"Sending request to Ollama API with model: {self.model} (chunk length: {len(prompt.split())} words)")
                response = requests.post(
                    f"{self.api_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "temperature": temperature,
                        "stream": False
                    },
                    timeout=self.timeout
                )
                
                if response.status_code != 200:
                    error_msg = f"Ollama API returned status code {response.status_code}"
                    try:
                        error_msg += f": {response.json()}"
                    except:
                        error_msg += f": {response.text}"
                    logging.error(error_msg)
                    raise Exception(error_msg)
                    
                response_data = response.json()
                if "error" in response_data:
                    error_msg = f"Ollama API returned error: {response_data['error']}"
                    logging.error(error_msg)
                    raise Exception(error_msg)
                    
                if "response" not in response_data:
                    error_msg = "Ollama API response missing 'response' field"
                    logging.error(error_msg)
                    raise Exception(error_msg)
                    
                responses.append(response_data["response"])
            
            # Combine responses if there were multiple chunks
            return "\n\n".join(responses)
            
        except requests.Timeout:
            error_msg = f"Ollama API request timed out after {self.timeout} seconds"
            logging.error(error_msg)
            raise Exception(error_msg)
        except requests.RequestException as e:
            error_msg = f"Ollama API request failed: {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error while calling Ollama API: {str(e)}"
            logging.error(error_msg)
            raise

def get_ai_service() -> AIService:
    service_type = os.environ.get("AI_SERVICE_TYPE", "openai").lower()
    if service_type == "ollama":
        return OllamaService()
    return OpenAIService()

app = Flask(__name__)
gitlab_token = os.environ.get("GITLAB_TOKEN")
gitlab_url = os.environ.get("GITLAB_URL")
ai_service = get_ai_service()

@app.route('/webhook', methods=['POST'], strict_slashes=False)
def webhook():
    logging.info(f"Received webhook request with headers: {dict(request.headers)}")
    
    if request.headers.get("X-Gitlab-Token") != os.environ.get("EXPECTED_GITLAB_TOKEN"):
        logging.warning("Unauthorized webhook request - invalid token")
        return "Unauthorized", 403
    
    payload = request.json
    logging.info(f"Webhook payload: {json.dumps(payload, indent=2)}")
    if payload.get("object_kind") == "merge_request":
        if payload["object_attributes"]["action"] != "open":
            return "Not a  PR open", 200
        project_id = payload["project"]["id"]
        mr_id = payload["object_attributes"]["iid"]
        changes_url = f"{gitlab_url}/projects/{project_id}/merge_requests/{mr_id}/changes"

        headers = {"Private-Token": gitlab_token}
        response = requests.get(changes_url, headers=headers)
        mr_changes = response.json()

        diffs = [change["diff"] for change in mr_changes["changes"]]

        # Prepare the diffs with line numbers
        numbered_diffs = []
        for change in mr_changes['changes']:
            diff_lines = change['diff'].split('\n')
            file_path = change['new_path']
            numbered_diffs.append(f"\n### Файл: {file_path}\n")
            for line in diff_lines:
                if line.startswith('+') or line.startswith('-'):
                    numbered_diffs.append(line)

        messages = [
            {"role": "system", "content": "You are a senior developer reviewing code changes. Provide detailed line-by-line analysis in Russian."},
            {"role": "user", "content": f"{MR_PRE_PROMPT}\n\n{''.join(numbered_diffs)}\n\n{MR_LINE_ANALYSIS_PROMPT}"},
            {"role": "assistant", "content": "Format the response in GitLab-friendly markdown. Provide line-specific comments and a brief summary at the end."}
        ]

        try:
            raw_answer = ai_service.generate_review(messages, temperature=0.2)
            
            # Format the response with proper structure
            sections = raw_answer.split('\n\n')
            line_comments = '\n'.join(sections[:-1])  # All sections except the last one are line comments
            summary = sections[-1]  # Last section is the summary
            
            answer = f"""## Анализ изменений по строкам

{line_comments}

## Краткое резюме
{summary}

---
Комментарий сгенерирован AI помощником для code review."""
        except Exception as e:
            print(e)
            answer = "I'm sorry, I'm not feeling well today. Please ask a human to review this PR."
            answer += "\n\nError: " + str(e)

        print(answer)
        comment_url = f"{gitlab_url}/projects/{project_id}/merge_requests/{mr_id}/notes"
        comment_payload = {"body": answer}
        comment_response = requests.post(comment_url, headers=headers, json=comment_payload)
    elif payload.get("object_kind") == "push":
        project_id = payload["project_id"]
        commit_id = payload["after"]
        commit_url = f"{gitlab_url}/projects/{project_id}/repository/commits/{commit_id}/diff"

        headers = {"Private-Token": gitlab_token}
        response = requests.get(commit_url, headers=headers)
        changes = response.json()

        changes_string = ''.join([str(change) for change in changes])

        messages = [
            {"role": "system", "content": "You are a senior developer reviewing code changes from a commit."},
            {"role": "user", "content": f"{COMMIT_PRE_PROMPT}\n\n{changes_string}{COMMIT_QUESTIONS}"},
            {"role": "assistant", "content": "Respond in markdown for GitLab. Include concise versions of questions in the response."},
        ]

        print(messages)
        try:
            answer = ai_service.generate_review(messages, temperature=0.7)

        except Exception as e:
            print(e)
            answer = "I'm sorry, I'm not feeling well today. Please ask a human to review this code change."
            answer += "\n\nError: " + str(e)

        print(answer)
        comment_url = f"{gitlab_url}/projects/{project_id}/repository/commits/{commit_id}/comments"
        comment_payload = {"note": answer}
        comment_response = requests.post(comment_url, headers=headers, json=comment_payload)

    return "OK", 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
