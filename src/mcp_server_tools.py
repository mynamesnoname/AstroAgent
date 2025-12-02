import asyncio
from typing import Any, List, Union, Dict
from mcp.server.fastmcp import FastMCP
from src.tools import _calculate_redshift, _predict_obs_wavelength, _galaxy_weighted_average_with_error, _QSO_rms

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

# @server.tool()
# def galaxy_weighted_average(
#     wavelength_obs: List[float],
#     wavelength_rest: List[float],
#     flux: List[float],
#     a: float,            # wavelength per pixel (Å/pix)
#     tolerance: int,      # 像素容差 t
#     rms_lambda: float    # 拟合波长 rms
# ) -> Dict:
#     """
#     MCP Tool: 计算 flux 加权平均红移及综合误差
#     """
#     try:
#         result = _galaxy_weighted_average_with_error(
#             wavelength_obs, 
#             wavelength_rest, 
#             flux, 
#             a, 
#             tolerance, 
#             rms_lambda
#         )
#         return result
#     except Exception as e:
#         print(f"error: {str(e)}")
#         return {"error": str(e)}
    
@server.tool()
def QSO_rms(wavelength_rest: float, 
    a: float,            # wavelength per pixel (Å/pix)
    tolerance: int,      # 像素容差 t
    rms_lambda: float    # 拟合波长 rms
    ):
    try:
        result = _QSO_rms(
            wavelength_rest, 
            a, 
            tolerance, 
            rms_lambda
        )
        return result
    except Exception as e:
        print(f"error: {str(e)}")
        return {"error": str(e)}



if __name__ == "__main__":
    print("[MCP Server] Spectro Tools Server is starting...")
    
    server.run(transport="stdio")
