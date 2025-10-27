import os
from abc import ABC, abstractmethod
import json
import re
from openai import RateLimitError
import asyncio

from .mcp_manager import MCPManager
from .utils import user_query

class BaseAgent(ABC):
    """基础智能体类 - 所有智能体的基类"""
    
    def __init__(self, agent_name: str, mcp_manager: MCPManager):
        self.agent_name = agent_name
        self.mcp_manager = mcp_manager
        
        # 延迟创建智能体实例，等到MCP工具初始化完成后再创建
        self.agent = None
        
        print(f"智能体 {agent_name} 初始化完成")

    async def ensure_agent_created(self):
        """确保智能体实例已创建（在MCP工具初始化后调用）"""
        if self.agent is None:
            self.agent = await self.mcp_manager.create_agent_with_tools(self.agent_name)
            print(f"智能体 {self.agent_name} 实例创建完成")

    async def call_llm_with_context(self, prompt, image_path=None, parse_json=True, description="LLM输出"):
        """
        调用 LLM 并可选直接解析 JSON。
        增强版：自动处理 RateLimitError 和 insufficient_quota。
        """
        max_retries = 3
        retry_delay = 180  # 3分钟

        for attempt in range(max_retries + 1):
            try:
                await self.ensure_agent_created()

                # --- 调用 ---
                if image_path:
                    message = user_query(prompt, image_path)
                    response = await self.agent['vis'].ainvoke({'messages': message}, config={"recursion_limit": 75})
                else:
                    message = user_query(prompt)
                    response = await self.agent['text'].ainvoke({'messages': message}, config={"recursion_limit": 75})

                raw_content = response['messages'][-1].content

                # --- 可选 JSON 解析 ---
                if parse_json:
                    cleaned = re.sub(r"^```(json)?|```$", "", raw_content.strip(), flags=re.IGNORECASE).strip()
                    try:
                        return json.loads(cleaned)
                    except json.JSONDecodeError:
                        raise ValueError(f"❌ {description} 非 JSON 或解析失败:\n{raw_content}")

                return raw_content

            # --- 限流错误处理 ---
            except RateLimitError as e:
                if attempt < max_retries:
                    print(f"⚠️ 智能体 {self.agent_name} - {description} 触发限流错误，{retry_delay // 60}分钟后重试...（第 {attempt + 1}/{max_retries} 次）")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    raise RuntimeError(f"❌ 智能体 {self.agent_name} - 超过最大重试次数，RateLimitError: {str(e)}")

            # --- 其他错误处理 ---
            except Exception as e:
                err_msg = str(e)
                if "insufficient_quota" in err_msg or "quota" in err_msg.lower():
                    print(f"❌ 智能体 {self.agent_name} - 账户额度已用尽，请检查充值或计费计划。原始错误信息：\n{err_msg}")
                    raise RuntimeError("调用失败：额度不足，请检查 DashScope / 通义千问 的配额状态。")
                elif "429" in err_msg:
                    print(f"⚠️ 智能体 {self.agent_name} - HTTP 429 错误，可能是限流或额度问题：\n{err_msg}")
                    if attempt < max_retries:
                        await asyncio.sleep(retry_delay)
                        continue
                else:
                    print(f"❌ 智能体 {self.agent_name} - 未知错误: {err_msg}")
                raise

        
        # async def call_llm_with_context(self, prompt, image_path=None):
        #     try:
        #         # 确保智能体实例已创建
        #         await self.ensure_agent_created()
        #         # 将系统和上下文组合成一个系统消息
        #         if image_path:
        #             message = user_query(prompt, image_path)
        #             response = await self.agent['vis'].ainvoke({'messages': message}, config={"recursion_limit": 75})
        #         else: 
        #             message = user_query(prompt)
        #             response = await self.agent['text'].ainvoke({'messages': message}, config={"recursion_limit": 75})
        #         return response['messages'][-1].content
        #     except Exception as e:
        #         error_msg = f"LLM调用失败: {str(e)}"
        #         print(f"❌ 智能体 {self.agent_name} - {error_msg}")
