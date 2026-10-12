
import requests

from config.settings import LLM_CONFIG
from utils.logger import logger, retry, log_time

class LLMClient:

    def __init__(self, config: dict = None):
        cfg = config or LLM_CONFIG
        self.base_url = cfg["base_url"]
        self.api_key = cfg["api_key"]
        self.model_name = cfg["model_name"]
        self.temperature = cfg["temperature"]
        self.max_tokens = cfg["max_tokens"]

        self._session = requests.Session()
        self._session.trust_env = False

    def _build_headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    @retry(max_times=3, delay=1.0)
    @log_time
    def chat(self, messages: list) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        resp = self._session.post(url, headers=self._build_headers(),
                                  json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        content = data["choices"][0]["message"]["content"]
        return content

    def simple_chat(self, user_input: str) -> str:
        messages = [{"role": "user", "content": user_input}]
        return self.chat(messages)

if __name__ == "__main__":

    client = LLMClient()
    if "在这里填" in client.api_key:
        logger.error("请先在 config/settings.py 里填入你的豆包 API Key")
    else:
        reply = client.simple_chat("你好，请用一句话介绍你自己")
        print("模型回复：", reply)
