# mcp_manager.py
import os
import json
import logging
from typing import Dict, Any, Optional
from langchain_mcp_adapters.client import MultiServerMCPClient

from AstroAgent.core.config.mcp_config import MCPConfig

class MCPManager:
    """
    MCP工具管理器 - 负责MCP客户端连接、工具发现和生命周期管理。

    Attributes
    ----------
    client: Optional[MultiServerMCPClient]
    """

    def __init__(self):

        self.client: Optional[MultiServerMCPClient] = None


    async def create_mcp_client(self, mcp_config: Dict[str, Any]) -> bool:
        """
        异步初始化 MCP 客户端

        Returns
        -------
        bool
            初始化成功返回 True

        Raises
        ------
        RuntimeError
            初始化失败时 raise
        """
        self.config: MCPConfig = mcp_config.config
        print(f"✅ MCPManager 初始化完成，使用 MCP 配置: {mcp_config.path}")
        print(f"✅ Done initializing MCPManager")

        if self.client:
            await self.close()

        try:
            self.client = MultiServerMCPClient(self.config)
            tools = await self.client.get_tools()
            print(f"[LLM INIT] 获取到的工具列表: {tools}")
            print(f"[LLM INIT] Done getting tools: {tools}")
            print("✅ MCP 客户端初始化成功")
            print("✅ Done initializing MCP client")
            return self.client
        except Exception as e:
            self.client = None
            raise RuntimeError(f"MCP 客户端初始化失败 Failed to initialize MCP client: {e}") from e


    async def close(self) -> None:
        """
        安全关闭 MCP 客户端
        """
        if not self.client:
            return

        try:
            if hasattr(self.client, 'close'):
                await self.client.close()
                logging.error("✅ MCP 客户端已关闭")
                logging.error("✅ Done closing MCP client")
            else:
                logging.error("ℹ️ MCP 客户端无需显式关闭")
                logging.error("ℹ️ No need to close MCP client explicitly")
        except Exception as e:
            logging.error(f"⚠️ 关闭 MCP 客户端时出错: {e}")
            logging.error("⚠️ Error closing MCP client: {e}")
        finally:
            self.client = None
