import os
import json
from typing import Optional, Dict, List
from pydantic import BaseModel, Field


# ------------------------
# Prompt 路径配置
# ------------------------

# 定义一个prompt配置类
class PromptFunctionConfig(BaseModel):
    """
    path: 每个agent的每个function所涉及的prompt路径文件夹
    variables: List[str] = [] # prompt 内所涉及的变量
    """
    path: str
    variables: List[str] = Field(default_factory=list)


class PromptConfig(BaseModel):
    info: Dict[str, Dict[str, PromptFunctionConfig]]
    root: str

    @classmethod
    def from_json(cls) -> "PromptConfig":
        prompt_root = os.getenv("PROMPTS_ROOT")
        path = os.getenv("PROMPTS_CONFIG_PATH")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(info=data, root=prompt_root)

    def get_function(self, agent_name: str, function_name: str) -> PromptFunctionConfig:
        return self.info[agent_name][function_name]
