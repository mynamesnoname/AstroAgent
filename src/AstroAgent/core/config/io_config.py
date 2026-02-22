import os
from pydantic import BaseModel


# ------------------------
# IO 配置（不含 batch）
# ------------------------

class IOConfig(BaseModel):
    input_dir: str
    output_dir: str
    run_mode: str
    image_name: str

    @classmethod
    def from_env(cls) -> "IOConfig":
        input_dir=os.getenv("INPUT_DIR")
        output_dir=os.getenv("OUTPUT_DIR")
        run_mode=(os.getenv("RUN_MODE") or "s").lower()
        image_name=os.getenv("IMAGE_NAME")
        
        if not all([input_dir, output_dir, run_mode, image_name]):
            raise ValueError("IO 配置 (input_dir/output_dir/run/image_name) 不完整")
        return cls(
            input_dir=input_dir,
            output_dir=output_dir,
            run_mode=run_mode,
            image_name=image_name,
        )
    
