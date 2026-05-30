"""ScholarAgent MCP Server entry point.

Usage:
    python main.py --mode server --transport stdio
    python main.py --mode list-tools
"""

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scholar_agent.core.logging import configure_logging, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="ScholarAgent MCP Server")
    parser.add_argument(
        "--mode",
        choices=["server", "list-tools"],
        default="server",
        help="server: start MCP service; list-tools: list registered tools and exit",
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    return parser.parse_args()


def main():
    log_file = configure_logging()
    logger = get_logger(__name__)
    args = parse_args()
    logger.info("ScholarAgent MCP entrypoint mode=%s transport=%s log_file=%s", args.mode, args.transport, log_file)
    from scholar_agent.mcp_server import ScholarMCPServer

    server = ScholarMCPServer()

    if args.mode == "list-tools":
        print("Registered MCP tools:")
        for name in server.get_registered_tool_names():
            print(f"  - {name}")
        return

    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
