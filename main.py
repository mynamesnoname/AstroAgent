#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
import logging
import csv
import numpy as np
from dotenv import load_dotenv
from src.workflow_orchestrator import WorkflowOrchestrator


async def main():
    """主异步入口：支持单图或批量分析"""
    try:
        load_dotenv()

        mcp_configs = os.getenv('MCP_CONFIG', '/home/wbc/code3/llm-spectro-agent/mcp_config.json')
        input_dir = os.getenv('INPUT_DIR')
        output_dir = os.getenv('OUTPUT_DIR', os.path.join(input_dir, "../output"))
        single_run = os.getenv('SINGLE_RUN', 'true').lower() == 'true'

        image_name = os.getenv('IMAGE_NAME')
        image_header = os.getenv('IMAGE_NAME_HEADER', '')
        # start = int(os.getenv('START', 0))
        # end = int(os.getenv('END', 0))

        if not input_dir:
            logging.error("❌ INPUT_DIR 未设置")
            return
        if not output_dir:
            logging.error("❌ OUTPUT_DIR 未设置")
            return
        if not os.path.isdir(input_dir):
            logging.error(f"❌ 输入目录不存在: {input_dir}")
            return
        if not os.path.isdir(output_dir):
            logging.error(f"输出目录不存在，已自动创建 {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        logging.info(f"✅ 使用配置文件: {mcp_configs}")

        orc = WorkflowOrchestrator(config_file=mcp_configs)
        await orc.initialize()

        if single_run:
            # 单张图像模式
            if not image_name:
                logging.error("❌ 单图模式下 IMAGE_NAME 未设置")
                return
            logging.info(f"🚀 开始单图分析: {image_name}.png")
            result = await orc.run_analysis_single()
            logging.info("✅ 单图分析完成")
        else:
            # 批量模式
            def parse_image_range(start_str, end_str):
                """解析起始和结束值，智能处理前导零"""
                start_num = int(start_str)
                end_num = int(end_str)
                
                # 如果原始字符串有前导零，保留格式
                if start_str.startswith('0') and end_str.startswith('0'):
                    width = max(len(start_str), len(end_str))
                    return start_num, end_num, width
                else:
                    return start_num, end_num, None

            start_str = os.getenv('START', '0')
            end_str = os.getenv('END', '0')

            start, end, width = parse_image_range(start_str, end_str)
            print(start, end, width)

            collect = []
            for i in range(start, end + 1):
                if width:
                    img_name = f"{image_header}{i:0{width}d}"
                else:
                    img_name = f"{image_header}{i}"
                
                os.environ['IMAGE_NAME'] = img_name
                image_path = os.path.join(input_dir, f"{img_name}.png")
                logging.info(f"🚀 批量分析模式: {start} → {end}")

            # for i in range(start, end + 1):
            #     img_name = f"{image_header}{i}"
            #     os.environ['IMAGE_NAME'] = img_name  # 临时覆盖

                # image_path = os.path.join(input_dir, f"{img_name}.png")
                
                # 检查文件是否存在
                if not os.path.isfile(image_path):
                    logging.warning(f"⚠️ 跳过：文件不存在 - {image_path}")
                    continue  # 批量模式跳过当前文件
                
                logging.info(f"➡️ 开始分析第 {i - start + 1}/{end - start + 1} 张图像: {img_name}.png")

                try:
                    result = await orc.run_analysis_single()
                    logging.info(f"✅ 图像 {img_name}.png 分析完成")

                    in_brief = result.get('in_brief', {})
                    def safe_str(x):
                        print('yes')
                        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
                            return 'N/A'
                        return str(x)

                    detail = [
                        img_name,
                        safe_str(in_brief.get('type_with_absention')),
                        safe_str(in_brief.get('type_forced')),
                        safe_str(in_brief.get('redshift')),
                        safe_str(in_brief.get('rms')),
                        safe_str(in_brief.get('lines')),
                        safe_str(in_brief.get('score')),
                        safe_str(in_brief.get('human'))
                    ]
                    collect.append(detail)
                except Exception as e:
                    logging.exception(f"❌ 图像 {img_name}.png 分析失败: {e}")

            csv_path = os.path.join(output_dir, 'in_brief.csv')
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['image_name', 'type_with_absention', 'type_forced', 'redshift', 'rms', 'lines', 'score', 'human'])
                writer.writerows(collect)

        logging.info("🎉 所有任务已完成")

    except Exception as e:
        logging.exception("❌ 主程序运行失败: %s", e)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    asyncio.run(main())
