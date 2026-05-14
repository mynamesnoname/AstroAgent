import logging
import functools

from dotenv import load_dotenv

from langchain_core.tools import BaseTool as LangChainBaseTool

from AstroAgent.manager.runtime.prompt_manager import PromptManager
from AstroAgent.manager.mcp.mcp_manager import MCPManager
from AstroAgent.core.llm import ThisIsModel

# ── MCP 工具超时/网络异常关键字（用于兜底字符串匹配） ──
_TIMEOUT_ERROR_KEYWORDS = (
    "readtimeout",
    "read timeout",
    "connecttimeout",
    "connect timeout",
    "timed out",
    "remote protocol error",
    "remoteprotocolerror",
)

# ── LLM 友好的超时提示模板 ──
_TIMEOUT_FEEDBACK_TEMPLATE = (
    "⚠️ MCP工具 '{tool_name}' 调用超时: {error_detail}\n"
    "这可能是由于计算量过大或网络波动导致。"
    "你可以：(1) 重试相同调用 (2) 简化请求参数 (3) 跳过此步骤继续分析。"
)


def _wrap_tool_for_timeout(tool):
    """包装 MCP 工具的 _arun，将超时异常转为 LLM 可见的错误消息。

    这样 LangGraph ToolNode 会将超时信息作为 ToolMessage 返回给 LLM，
    LLM 可以基于上下文自主决策（重试/跳过/调整参数），
    而不是由外层机械重试整个调用。
    """
    # 仅包装覆盖了异步实现的工具
    if type(tool)._arun is not LangChainBaseTool._arun:
        original_arun = tool._arun
        tool_name = tool.name

        @functools.wraps(original_arun)
        async def _arun_wrapper(*args, **kwargs):
            try:
                return await original_arun(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                if any(kw in error_str for kw in _TIMEOUT_ERROR_KEYWORDS):
                    logging.warning(
                        f"⏱️ MCP工具 '{tool_name}' 超时，将反馈给 LLM: {str(e)[:150]}"
                    )
                    return _TIMEOUT_FEEDBACK_TEMPLATE.format(
                        tool_name=tool_name,
                        error_detail=str(e)[:300],
                    )
                raise

        tool._arun = _arun_wrapper

    return tool


class RuntimeContainer:

    def __init__(self, configs):
        self.configs = configs
        self._mcp = None
        self._tools = None
        self._models = {}
        self.prompt_manager = PromptManager(self.configs.prompt)

    async def _create_mcp_client(self):
        manager = MCPManager()
        self._mcp = await manager.create_mcp_client(
            self.configs.mcp
        )

    async def _ensure_mcp(self):
        if self._mcp is None:
            manager = MCPManager()
            self._mcp = await manager.create_mcp_client(
                self.configs.mcp
            )

    async def reset_mcp(self) -> None:
        """断网重联时调用：清除缓存的 MCP 连接和工具列表，下次 get_tools() 时自动重建。"""
        logging.warning("🔄 重置 MCP 连接缓存，等待重新连接...")
        self._tools = None
        self._mcp = None

    async def get_tools(self):
        if self._tools is None:
            await self._ensure_mcp()
            raw_tools = await self._mcp.get_tools()
            # 包装每个 MCP 工具，将超时异常转为 LLM 可理解的错误消息
            self._tools = [_wrap_tool_for_timeout(t) for t in raw_tools]
        return self._tools

    def get_model(self, type):
        if type not in self._models:
            config = self.configs.model.__dict__[type]
            self._models[type] = ThisIsModel(config).create_client()
        return self._models[type]
