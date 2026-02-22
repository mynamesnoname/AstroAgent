import os
import json
import logging
from typing import List, Dict, Any
from pydantic import BaseModel

# ------------------------
# MCP 配置
# ------------------------

class MCPConfig(BaseModel):
    path: str  # 配置文件路径
    config: Dict[str, Any]

    @classmethod
    def from_env(cls) -> "MCPConfig":
        path = os.getenv("MCP_CONFIG")  # 从环境变量中获取配置路径
        if not path:
            raise ValueError("环境变量 'MCP_CONFIG' 未设置，请检查配置。")
        
        # 调用类方法加载配置文件
        config = cls._load_config(path)
        
        # 创建并返回 MCPConfig 实例
        return cls(path=path, config=config)  # 将字典解包并传递给 MCPConfig

    @classmethod
    def _load_config(cls, config_file: str) -> Dict[str, Any]:
        """
        加载 MCP JSON 配置文件，文件不存在或解析失败直接 raise

        Parameters
        ----------
        config_file : str
            MCP 配置文件路径

        Returns
        -------
        dict
            配置内容字典
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"MCP 配置文件不存在: {config_file}")

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logging.info(f"✅ MCP 配置文件加载成功: {config_file}")
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"MCP 配置文件解析失败: {config_file}, error: {e}") from e
