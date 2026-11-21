"""indian-stocks-mcp — MCP server for India-only (NSE/BSE) stock analysis."""

from __future__ import annotations

__all__ = ["main", "__version__"]

__version__ = "0.3.0"


def main() -> None:
    """Entry point.

    ``server`` is imported lazily: importing it eagerly at package import
    time makes ``python -m stock_analyst_mcp.server`` emit
    ``RuntimeWarning: 'stock_analyst_mcp.server' found in sys.modules`` (the
    module would already be imported as a dependency of the package before
    runpy executes it). Laziness keeps that path warning-free.
    """
    from .server import main as _main

    _main()
