import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI

class BaseLLM():
    """
    LLM initialization
    """
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config

class ThisIsModel(BaseLLM):
    """
    LLM/VLM initialization
    """
    def create_client(self):
        try:
            # 尝试创建 ChatOpenAI 客户端
            client = ChatOpenAI(
                model=self.model_config['model'],
                api_key=self.model_config['api_key'],
                base_url=self.model_config['base_url'],
                temperature=self.model_config['temperature'],
                max_tokens=self.model_config['max_tokens'],
            )
            return client
        except KeyError as e:
            # 捕获缺少关键配置项的错误
            error_message = f"Configuration key missing: {str(e)}"
            logging.error(f"LLM Client creation failed: {error_message}")
            raise ValueError(f"LLM Client creation failed: {error_message}") from e
        except Exception as e:
            # 捕获其他错误
            logging.error(f"LLM Client creation failed: {str(e)}")
            raise RuntimeError(f"LLM Client creation failed: {str(e)}") from e
