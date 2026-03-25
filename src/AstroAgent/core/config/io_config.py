import os
from pydantic import BaseModel


# ------------------------
# IO 配置（不含 batch）
# ------------------------

class IOConfig(BaseModel):
    input_dir: str
    input_format: str
    output_dir: str
    run_mode: str
    file_name: str

    @classmethod
    def from_env(cls) -> "IOConfig":
        input_dir=os.getenv("INPUT_DIR")
        input_format=os.getenv("INPUT_FORMAT")
        output_dir=os.getenv("OUTPUT_DIR")
        run_mode=(os.getenv("RUN_MODE") or "s").lower()
        file_name=os.getenv("FILE_NAME")
        
        if not all([input_dir, output_dir, run_mode, file_name]):
            raise ValueError("IO 配置 (input_dir/output_dir/run/file_name) 不完整")
        return cls(
            input_dir=input_dir,
            input_format=input_format,
            output_dir=output_dir,
            run_mode=run_mode,
            file_name=file_name,
        )
    
