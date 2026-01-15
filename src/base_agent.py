import os
from abc import ABC, abstractmethod
import json
import re
from openai import RateLimitError
import asyncio
import logging

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

    async def call_llm_with_context(self, system_prompt, user_prompt, image_path=None, parse_json=True, description="LLM输出", OCR=False):
        """
        调用 LLM 并可选直接解析 JSON。
        增强版：自动处理 RateLimitError 和 insufficient_quota。
        """
        max_retries = 3
        retry_delay = 180  # 3分钟

        for attempt in range(max_retries + 1):
            try:
                await self.ensure_agent_created()

                # --- 构建消息 ---
                messages = user_query(system_prompt, user_prompt, image_path)
                
                # --- 调用 ---
                if image_path:
                    response = await self.agent['vis'].ainvoke({'messages': messages}, config={"recursion_limit": 300})
                else:
                    response = await self.agent['text'].ainvoke({'messages': messages}, config={"recursion_limit": 125})

                raw_content = response['messages'][-1].content

                if parse_json:
                    # 清理可能的 markdown code block 包裹
                    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content.strip(), flags=re.IGNORECASE).strip()
                    
                    try:
                        # 尝试解析为 JSON
                        parsed = json.loads(cleaned)
                        logging.debug("✅ JSON 解析成功")
                        return parsed
                    except (json.JSONDecodeError, TypeError) as e:
                        # 解析失败 → 降级：返回原始字符串（或 cleaned）
                        # logging.warning(f"⚠️ JSON 解析失败，降级为字符串: {str(e)[:100]}...")
                        # 可选：记录原始内容片段用于调试
                        # logging.debug(f"Raw content preview: {repr(cleaned[:200])}")
                        return cleaned  # 或 return raw_content，看你偏好
                else:
                    return raw_content
                    
            except Exception as e:
                # 原有的异常处理逻辑保持不变
                error_msg = str(e).lower()
                if attempt < max_retries and ("rate limit" in error_msg or "insufficient_quota" in error_msg):
                    logging.warning(f"⏳ {description}遇到限制，{retry_delay}秒后重试... (尝试 {attempt + 1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                else:
                    logging.error(f"❌ {description}失败: {str(e)}")
                    raise