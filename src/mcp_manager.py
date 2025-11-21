import os
import json
from typing import Dict, Any, Optional
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
# from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


class MCPManager:
    """MCP工具管理器 - 负责MCP连接、工具发现和权限控制"""

    def __init__(self, config_file: str):
        load_dotenv()
        self.config = self._load_config(config_file)
        self.llm, self.vis_llm = self._init_llm()
        self.client: Optional[MultiServerMCPClient] = None
        print("✅ MCP管理器初始化完成")

    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """加载 JSON 配置文件，失败时返回默认空配置"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✅ 配置文件加载成功: {config_file}")
            return config
        except FileNotFoundError:
            print(f"⚠️ 配置文件未找到: {config_file}，使用默认配置")
        except json.JSONDecodeError as e:
            print(f"❌ 配置文件格式错误: {e}")
        return {"servers": {}, "agent_permissions": {}}

    def _get_env_or_raise(self, key: str, default: Optional[str] = None) -> str:
        """从环境变量获取值，若缺失且无默认值则报错"""
        value = os.getenv(key, default)
        if value is None:
            raise ValueError(f"环境变量 {key} 未设置，请检查 .env 文件")
        return value.strip()

    def _init_llm(self):
        """初始化文本和视觉大模型"""
        # 公共配置解析逻辑
        def _build_llm(llm_type: str, default_model: str):
            api_key = self._get_env_or_raise(f"{llm_type}_API_KEY")
            base_url = self._get_env_or_raise(f"{llm_type}_BASE_URL").rstrip()
            model = os.getenv(f"{llm_type}_MODEL", default_model)
            temp_str = os.getenv(f"{llm_type}_TEMPERATURE", "0.1")
            temperature = float(temp_str) if temp_str else 0.1
            max_tokens_str = os.getenv(f"{llm_type}_MAX_TOKENS")
            max_tokens = int(max_tokens_str) if max_tokens_str else None

            print(f"[LLM INIT] {llm_type} -> model={model}, temp={temperature}, max_tokens={max_tokens}, base_url={base_url}")
            return ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        llm = _build_llm("LLM", "qwen3-max-2025-09-23")
        vis_llm = _build_llm("VIS_LLM", "qwen-vl-max-2025-08-13")
        return llm, vis_llm

    async def initialize(self):
        """初始化 MCP 客户端并构建 agents"""
        if self.client:
            await self.close()
        try:
            self.client = MultiServerMCPClient(self.config)
            print(f"✅ MCP 客户端初始化成功")
            return True
        except Exception as e:
            print(f"❌ MCP 客户端初始化失败: {e}")
            self.client = None
            return False
    
    async def create_agent_with_tools(self, agent_name):
        if agent_name == 'Spectral Visual Interpreter':
            tools = []
        else:
            tools = await self.client.get_tools()
        text_agent = create_agent(self.llm, tools)
        vis_agent = create_agent(self.vis_llm, tools)
        agents = {"text": text_agent, "vis": vis_agent}
        return agents

    async def close(self):
        """安全关闭 MCP 客户端"""
        if not self.client:
            return

        try:
            if hasattr(self.client, 'close'):
                await self.client.close()
                print("✅ MCP 连接已关闭")
            else:
                print("ℹ️ MCP 客户端无需显式关闭")
        except Exception as e:
            print(f"⚠️ 关闭 MCP 客户端时出错: {e}")
        finally:
            self.client = None