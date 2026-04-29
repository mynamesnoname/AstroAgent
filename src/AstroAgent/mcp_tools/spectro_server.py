# mcp_server_tools.py
import os
from typing import List, Union
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

from tools import (
    calculate_redshift,
    predict_obs_wavelength,
    redshift_rms,
    calculate_peak_amplitude_ratio,
)
from tool_protocol import ToolResult, ToolError

_MCP_PORT = int(os.getenv("MCP_SERVER_PORT", "8765"))
server = FastMCP("spectro_tools", host="127.0.0.1", port=_MCP_PORT)


# @server.tool()
# def calculate_redshift_tool(
#     obs_wavelength: Union[float, List[float]],
#     rest_wavelength: Union[float, List[float]],
# ) -> dict:
#     """
#     Calculate redshift z from observed and rest-frame wavelengths.
#     """
#     try:
#         z = calculate_redshift(obs_wavelength, rest_wavelength)

#         return ToolResult(
#             success=True,
#             result={
#                 "quantity": "redshift",
#                 "symbol": "z",
#                 "values": z,
#                 "definition": "z = (λ_obs / λ_rest) - 1",
#                 "input_type": "scalar" if isinstance(z, float) else "array",
#             },
#         ).model_dump()

#     except Exception as e:
#         return ToolResult(
#             success=False,
#             error=ToolError(
#                 type="InvalidInput",
#                 message=str(e),
#                 hint="Provide scalar inputs or lists of equal length.",
#             ),
#         ).model_dump()


# @server.tool()
# def predict_obs_wavelength_tool(
#     redshift: Union[float, List[float]],
#     rest_wavelength: Union[float, List[float]],
# ) -> dict:
#     """
#     Predict observed wavelength from redshift and rest-frame wavelength.
#     """
#     try:
#         obs = predict_obs_wavelength(redshift, rest_wavelength)

#         return ToolResult(
#             success=True,
#             result={
#                 "quantity": "observed_wavelength",
#                 "symbol": "λ_obs",
#                 "values": obs,
#                 "unit": "Angstrom",
#                 "definition": "λ_obs = λ_rest · (1 + z)",
#                 "input_type": "scalar" if isinstance(obs, float) else "array",
#             },
#         ).model_dump()

#     except Exception as e:
#         return ToolResult(
#             success=False,
#             error=ToolError(
#                 type="InvalidInput",
#                 message=str(e),
#                 hint="Ensure redshift and rest_wavelength have compatible shapes.",
#             ),
#         ).model_dump()


@server.tool()
def calculate_rms_for_redshift_tool(
    wavelength_rest: float,
    wavelength_error: float,
) -> dict:
    """
    Calculate RMS uncertainty of redshift measurement for spectra.
    """
    try:
        rms = redshift_rms(
            wavelength_rest=wavelength_rest,
            wavelength_error=wavelength_error,
        )

        return ToolResult(
            success=True,
            result={
                "quantity": "redshift_rms",
                "symbol": "σ_z",
                "value": rms,
                "dimensionless": True,
                "definition": "σ_z = σ_λ / λ_rest",
            },
        ).model_dump()

    except Exception as e:
        return ToolResult(
            success=False,
            error=ToolError(
                type="RuntimeError",
                message=str(e),
            ),
        ).model_dump()


# @server.tool()
# def calculate_peak_amplitude_ratio_tool(
#     wavelength1: float,
#     amplitude1: float,
#     wavelength2: float,
#     amplitude2: float,
# ) -> dict:
#     """
#     Calculate the amplitude ratio of two spectral peaks.
    
#     The ratio is computed as: ratio = amplitude1 / amplitude2, where:
#     - amplitude1 is the amplitude of the first peak at wavelength1
#     - amplitude2 is the amplitude of the second peak at wavelength2
    
#     This tool is useful for analyzing double-peaked spectral features
#     commonly found in cosmological spectra (e.g., emission line doublets).
#     """
#     try:
#         ratio = calculate_peak_amplitude_ratio(
#             wavelength1=wavelength1,
#             amplitude1=amplitude1,
#             wavelength2=wavelength2,
#             amplitude2=amplitude2,
#         )

#         return ToolResult(
#             success=True,
#             result={
#                 "quantity": "amplitude_ratio",
#                 "symbol": "R",
#                 "value": ratio,
#                 "dimensionless": True,
#                 "definition": "R = amplitude1 / amplitude2",
#                 "peak_1": {
#                     "wavelength": wavelength1,
#                     "amplitude": amplitude1,
#                     "description": "Numerator peak (first peak)",
#                 },
#                 "peak_2": {
#                     "wavelength": wavelength2,
#                     "amplitude": amplitude2,
#                     "description": "Denominator peak (second peak)",
#                 },
#                 "interpretation": f"Peak at {wavelength1} Å is {ratio:.4f} times the amplitude of peak at {wavelength2} Å",
#             },
#         ).model_dump()

#     except Exception as e:
#         return ToolResult(
#             success=False,
#             error=ToolError(
#                 type="InvalidInput",
#                 message=str(e),
#                 hint="Ensure amplitude2 is not zero. Both amplitudes should be valid numerical values.",
#             ),
#         ).model_dump()

if __name__ == "__main__":
    # print("[MCP Server] Spectro Tools Server is starting...")
    # transport="streamable-http" enables concurrent-safe HTTP connections,
    # replacing stdio which serialises all requests through a single pipe.
    server.run(transport="streamable-http")
