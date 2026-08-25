"""Basic arithmetic tools exposed over MCP."""

from fastmcp import FastMCP

mcp = FastMCP("Calculator")


@mcp.tool
async def add(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return a + b


@mcp.tool
async def subtract(a: float, b: float) -> float:
    """Subtract b from a and return the difference."""
    return a - b


@mcp.tool
async def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the product."""
    return a * b


@mcp.tool
async def divide(a: float, b: float) -> float:
    """Divide a by b and return the quotient."""
    if b == 0:
        raise ValueError("cannot divide by zero")
    return a / b


if __name__ == "__main__":
    mcp.run()
