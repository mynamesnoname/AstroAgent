import asyncio
from typing import Any, List, Union
from mcp.server.fastmcp import FastMCP
from src.tools import _calculate_redshift, _predict_obs_wavelength

server = FastMCP("spectro_tools")

@server.tool()
def calculate_redshift(obs_wavelength: Union[float, List[float]], rest_wavelength: Union[float, List[float]]) -> dict:
    return _calculate_redshift(obs_wavelength, rest_wavelength)

@server.tool()
def predict_obs_wavelength(redshift: Union[float, List[float]], rest_wavelength: Union[float, List[float]]) -> dict:
    return _predict_obs_wavelength(redshift, rest_wavelength)

if __name__ == "__main__":
    print("[MCP Server] Spectro Tools Server is starting...")
    
    server.run(transport="stdio")
