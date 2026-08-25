"""One-time OAuth login for the hosted Expense Tracker MCP server.

Run this once:   python oauth_login.py

It opens a browser, completes the OAuth handshake over a single serialized
connection, then writes the access token into .env as EXPENSE_MCP_TOKEN.
chatbot_mcp.py then just reads that token -- it never runs OAuth itself.

Why not let MultiServerMCPClient do the OAuth? Because the streamable-HTTP
transport issues several requests against the same auth object, so the
registration used for the browser leg is not necessarily the one used for the
token exchange -- which is what produced `Token exchange failed (401)`.
"""

import asyncio
from pathlib import Path

from fastmcp import Client
from fastmcp.client.auth import OAuth

URL = "https://voiceless-chocolate-pike.fastmcp.app/mcp"
ENV_PATH = Path(__file__).with_name(".env")
ENV_KEY = "EXPENSE_MCP_TOKEN"


def write_env(key: str, value: str) -> None:
    """Insert or replace `key` in .env, leaving every other line untouched."""
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    for i, line in enumerate(lines):
        if line.split("=", 1)[0].strip() == key:
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main() -> None:
    # A fixed callback port keeps the redirect_uri stable across the flow.
    oauth = OAuth(mcp_url=URL, callback_port=8765)

    async with Client(URL, auth=oauth) as client:
        tools = await client.list_tools()
        print("Connected. Tools:", [t.name for t in tools])

    tokens = oauth.context.current_tokens
    if not tokens or not tokens.access_token:
        raise SystemExit("Handshake finished but no access token was returned.")

    write_env(ENV_KEY, tokens.access_token)
    print(f"\nWrote {ENV_KEY} to {ENV_PATH}")
    if tokens.expires_in:
        print(f"Token expires in {tokens.expires_in}s -- rerun this script when it lapses.")


if __name__ == "__main__":
    asyncio.run(main())
