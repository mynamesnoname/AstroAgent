# mcp_server_tools.py
from typing import List, Union
from mcp.server.fastmcp import FastMCP

from tools import (
    calculate_redshift,
    predict_obs_wavelength,
    qso_redshift_rms,
)
from tool_protocol import ToolResult, ToolError

server = FastMCP("spectro_tools")


@server.tool()
def calculate_redshift_tool(
    obs_wavelength: Union[float, List[float]],
    rest_wavelength: Union[float, List[float]],
) -> dict:
    """
    Calculate redshift z from observed and rest-frame wavelengths.
    """
    try:
        z = calculate_redshift(obs_wavelength, rest_wavelength)

        return ToolResult(
            success=True,
            result={
                "quantity": "redshift",
                "symbol": "z",
                "values": z,
                "definition": "z = (λ_obs / λ_rest) - 1",
                "input_type": "scalar" if isinstance(z, float) else "array",
            },
        ).model_dump()

    except Exception as e:
        return ToolResult(
            success=False,
            error=ToolError(
                type="InvalidInput",
                message=str(e),
                hint="Provide scalar inputs or lists of equal length.",
            ),
        ).model_dump()


@server.tool()
def predict_obs_wavelength_tool(
    redshift: Union[float, List[float]],
    rest_wavelength: Union[float, List[float]],
) -> dict:
    """
    Predict observed wavelength from redshift and rest-frame wavelength.
    """
    try:
        obs = predict_obs_wavelength(redshift, rest_wavelength)

        return ToolResult(
            success=True,
            result={
                "quantity": "observed_wavelength",
                "symbol": "λ_obs",
                "values": obs,
                "unit": "Angstrom",
                "definition": "λ_obs = λ_rest · (1 + z)",
                "input_type": "scalar" if isinstance(obs, float) else "array",
            },
        ).model_dump()

    except Exception as e:
        return ToolResult(
            success=False,
            error=ToolError(
                type="InvalidInput",
                message=str(e),
                hint="Ensure redshift and rest_wavelength have compatible shapes.",
            ),
        ).model_dump()


@server.tool()
def calculate_rms_for_qso_redshift_tool(
    wavelength_rest: float,
    a: float,
    tolerance: int,
    rms_lambda: float,
) -> dict:
    """
    Calculate RMS uncertainty of redshift measurement for QSO spectra.
    """
    try:
        rms = qso_redshift_rms(
            wavelength_rest=wavelength_rest,
            a=a,
            tolerance=tolerance,
            rms_lambda=rms_lambda,
        )

        return ToolResult(
            success=True,
            result={
                "quantity": "redshift_rms",
                "symbol": "σ_z",
                "value": rms,
                "dimensionless": True,
                "definition": "σ_z = sqrt((a·t)^2 + σ_λ^2) / λ_rest",
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

if __name__ == "__main__":
    # print("[MCP Server] Spectro Tools Server is starting...")
    server.run(transport="stdio")
