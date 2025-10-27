import asyncio
from typing import Any, List, Union
from mcp.server.fastmcp import FastMCP
from src.tools import _calculate_redshift, _predict_obs_wavelength, _weighted_average_with_error

server = FastMCP("spectro_tools")

@server.tool()
def calculate_redshift(obs_wavelength: Union[float, List[float]], rest_wavelength: Union[float, List[float]]) -> dict:
    return _calculate_redshift(obs_wavelength, rest_wavelength)

@server.tool()
def predict_obs_wavelength(redshift: Union[float, List[float]], rest_wavelength: Union[float, List[float]]) -> dict:
    return _predict_obs_wavelength(redshift, rest_wavelength)

# @server.tool()
# def weighted_average(redshift: List[float], flux: List[float]) -> float:
#     return _weighted_average(redshift, flux)

@server.tool()
def weighted_average(redshift: List[float], flux: List[float]) -> dict:
    return _weighted_average_with_error(redshift, flux)


if __name__ == "__main__":
    print("[MCP Server] Spectro Tools Server is starting...")
    
    server.run(transport="stdio")
