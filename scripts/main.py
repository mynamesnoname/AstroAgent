import asyncio
import os
import logging
import csv
import numpy as np
from dotenv import load_dotenv

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from AstroAgent.core.config.all_config import AllConfig
from AstroAgent.workflow_orchestrator import WorkflowOrchestrator
from AstroAgent.manager.runtime.state_manager import SpectroStateFactory
from AstroAgent.agents.common.result_writer import ResultWriter


async def main():
    """Main asynchronous entry: unified single / batch image analysis"""
    try:
        load_dotenv()

        # ------------------------
        # Load configs
        # ------------------------
        configs = AllConfig.from_env()
        # model_config = configs.model
        io_config = configs.io
        batch_config = configs.batch
        # params_config = configs.params
        factory = SpectroStateFactory(configs)
        writer = ResultWriter()

        input_dir = io_config.input_dir
        output_dir = io_config.output_dir

        # ------------------------
        # Check directories
        # ------------------------
        if not input_dir:
            logging.error("INPUT_DIR is not set")
            return

        if not os.path.isdir(input_dir):
            logging.error(f"Input directory does not exist: {input_dir}")
            return

        if not output_dir:
            logging.error("OUTPUT_DIR is not set")
            return

        if not os.path.isdir(output_dir):
            logging.info(f"Output directory does not exist, creating: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        logging.info(f"Using MCP config: {configs.mcp.path}")

        # ------------------------
        # Resolve image IDs (single or batch)
        # ------------------------
        if io_config.run_mode=='b':
            if batch_config.is_batch_mode():
                file_ids = batch_config.generate_ids()
                logging.info(f"Batch mode enabled, total images: {len(file_ids)}")
            else: 
                raise ValueError("Batch mode is enabled but start/end are not set")
        else:
            if not io_config.file_name:
                logging.error("IMAGE_NAME is not set for single image mode")
                return
            file_ids = [io_config.file_name]
            logging.info("Single image mode enabled")

        # ------------------------
        # Initialize orchestrator
        # ------------------------
        orchestrator = WorkflowOrchestrator(configs)
        # await orchestrator.initialize()

        # ------------------------
        # Helper for safe string conversion
        # ------------------------
        def safe_str(x):
            if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
                return "N/A"
            return str(x)

        # ------------------------
        # Main processing loop
        # ------------------------
        results = []
        total = len(file_ids)

        for idx, file_name in enumerate(file_ids, start=1):
            format = io_config.input_format
            file_path = os.path.join(input_dir, f"{file_name}.{format}")

            if not os.path.isfile(file_path):
                logging.warning(f"Skipping missing file: {file_path}")
                continue

            logging.info(f"Processing image {idx}/{total}: {file_name}.{format}")

            try:
                state = factory.create(
                    file_name=file_name,
                    input_dir=input_dir,
                    output_dir=output_dir
                )
                result = await orchestrator.run_analysis_single(state)
                writer.write(result)
                logging.info(f"Image {file_name}.{format} processed")

                in_brief = result.get("in_brief", {})
                results.append([
                    file_name,
                    safe_str(in_brief.get("type_with_absention")),
                    safe_str(in_brief.get("type_forced")),
                    safe_str(in_brief.get("redshift")),
                    safe_str(in_brief.get("rms")),
                    safe_str(in_brief.get("lines")),
                    safe_str(in_brief.get("score")),
                    safe_str(in_brief.get("human")),
                ])

            except Exception as e:
                logging.exception(f"Failed to process image {file_name}.{format}: {e}")

        # ------------------------
        # Save results
        # ------------------------
        if results:
            csv_path = os.path.join(output_dir, "in_brief.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer_ = csv.writer(f)
                writer_.writerow([
                    "file_name",
                    "type_with_absention",
                    "type_forced",
                    "redshift",
                    "rms",
                    "lines",
                    "score",
                    "human",
                ])
                writer_.writerows(results)

            logging.info(f"Results saved to {csv_path}")
        else:
            logging.warning("No valid results to save")

        logging.info("All tasks completed")

    except Exception as e:
        logging.exception(f"Main program failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    asyncio.run(main())
