"""ScholarAgent Web Server entry point.

Usage:
    python run_web.py
"""

from pathlib import Path

import uvicorn
from uvicorn.supervisors import ChangeReload

from scholar_agent.config import conf
from scholar_agent.core.runtime import request_shutdown
from scholar_agent.core.logging import configure_logging, get_logger
from scholar_agent.mcp_client import shutdown_external_tools


class ScholarAgentServer(uvicorn.Server):
    """Uvicorn server that notifies worker loops as soon as Ctrl+C arrives."""

    def handle_exit(self, sig, frame) -> None:
        request_shutdown(f"signal {sig}")
        super().handle_exit(sig, frame)


def _split_reload_dirs(value: str) -> list[str]:
    raw_dirs = [part.strip() for part in str(value or "").replace(";", ",").split(",")]
    dirs = []
    for raw in raw_dirs:
        if not raw:
            continue
        dirs.append(str(Path(raw).resolve()))
    return dirs or [str(Path(conf.PROJECT_ROOT).resolve())]


if __name__ == "__main__":
    log_file = configure_logging()
    logger = get_logger(__name__)
    reload_dirs = _split_reload_dirs(conf.WEB_RELOAD_DIRS)
    logger.info(
        "starting ScholarAgent web server log_file=%s host=%s port=%s reload=%s reload_dirs=%s",
        log_file,
        conf.WEB_HOST,
        conf.WEB_PORT,
        conf.WEB_RELOAD,
        reload_dirs,
    )
    config = uvicorn.Config(
        "scholar_agent.web.server:app",
        host=conf.WEB_HOST,
        port=conf.WEB_PORT,
        reload=conf.WEB_RELOAD,
        reload_dirs=reload_dirs,
        reload_includes=["*.py", "*.html", "*.css", "*.js", "*.svg"],
        reload_excludes=[
            ".git/*",
            "__pycache__/*",
            "workspace/db/*",
            "workspace/logs/*",
            "workspace/models/*",
            "workspace/papers/*",
        ],
        log_config=None,
        timeout_graceful_shutdown=5,
    )
    server = ScholarAgentServer(config)
    try:
        if config.should_reload:
            sock = config.bind_socket()
            ChangeReload(config, target=server.run, sockets=[sock]).run()
        else:
            server.run()
    except KeyboardInterrupt:
        logger.info("ScholarAgent web server interrupted by Ctrl+C")
    finally:
        request_shutdown("run_web exiting")
        shutdown_external_tools()
