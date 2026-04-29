import os
import json
import logging
from typing import List, Dict, Any
from pydantic import BaseModel

# ------------------------
# MCP 配置
# ------------------------

# 默认 MCP server 监听地址（仅当 JSON 中未指定 url 时使用）
_DEFAULT_MCP_HOST = "127.0.0.1"
_DEFAULT_MCP_PORT = "8765"
_DEFAULT_MCP_PATH = "/mcp"

class MCPConfig(BaseModel):
    path: str  # 配置文件路径
    config: Dict[str, Any]
    server_script: str   # MCP server 脚本路径
    startup_timeout: int  # MCP server 启动超时（秒）

    @classmethod
    def from_env(cls) -> "MCPConfig":
        path = os.getenv("MCP_CONFIG")  # 从环境变量中获取配置路径
        if not path:
            raise ValueError("环境变量 'MCP_CONFIG' 未设置，请检查配置。")
        
        # 调用类方法加载配置文件
        config = cls._load_config(path)

        server_script = os.getenv("MCP_SERVER_SCRIPT", "")
        startup_timeout = int(os.getenv("MCP_STARTUP_TIMEOUT", "15"))
        
        # 创建并返回 MCPConfig 实例
        return cls(
            path=path,
            config=config,
            server_script=server_script,
            startup_timeout=startup_timeout,
        )

    @classmethod
    def _load_config(cls, config_file: str) -> Dict[str, Any]:
        """
        加载 MCP JSON 配置文件，文件不存在或解析失败直接 raise

        如果 JSON 中未指定 url，则自动从 MCP_SERVER_PORT 环境变量拼接，
        确保 server 端（spectro_server.py）和 client 端端口一致。

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
        except json.JSONDecodeError as e:
            raise ValueError(f"MCP 配置文件解析失败: {config_file}, error: {e}") from e

        # ── 统一注入 url：从 MCP_SERVER_PORT 环境变量拼接 ──
        mcp_port = os.getenv("MCP_SERVER_PORT", _DEFAULT_MCP_PORT)
        auto_url = f"http://{_DEFAULT_MCP_HOST}:{mcp_port}{_DEFAULT_MCP_PATH}"

        for _name, _cfg in config.items():
            if isinstance(_cfg, dict):
                _cfg["url"] = auto_url

        logging.info(f"✅ MCP 配置文件加载成功: {config_file}, url={auto_url}")
        return config
