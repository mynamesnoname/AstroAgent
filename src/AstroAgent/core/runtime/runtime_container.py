from dotenv import load_dotenv

from AstroAgent.manager.runtime.prompt_manager import PromptManager
from AstroAgent.manager.mcp.mcp_manager import MCPManager
from AstroAgent.core.llm import ThisIsModel


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

    async def get_tools(self):
        if self._tools is None:
            await self._ensure_mcp()
            self._tools = await self._mcp.get_tools()
        return self._tools

    def get_model(self, type):
        if type not in self._models:
            config = self.configs.model.__dict__[type]
            self._models[type] = ThisIsModel(config).create_client()
        return self._models[type]
