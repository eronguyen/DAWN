import logging
import os
from rich.logging import RichHandler
from rich.traceback import install
from rich.console import Console
from accelerate.logging import get_logger
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

def setup_logging(is_main_process: bool = True, log_dir: str = "outputs/") -> None:
    """Setup logging according to `training_args`."""
    os.makedirs(log_dir, exist_ok=True)
    # Rich ANSI log file.
    ansi_log_file = open(os.path.join(log_dir, f"log.ansi"), "w", encoding="utf-8")
    console_file = Console(file=ansi_log_file, force_terminal=True, width=180, record=True, stderr=True)
    ansi_handler = RichHandler(console=console_file, rich_tracebacks=False, show_path=False, markup=True)

    # Plain text log file.
    plain_handler = logging.FileHandler(os.path.join(log_dir, f"log.log"), mode="w", encoding="utf-8")
    plain_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
        )
    )

    # Rich console log for main process.
    rich_handler = RichHandler(rich_tracebacks=False, show_path=False, markup=True)
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    handlers = [ansi_handler, plain_handler]
    if is_main_process:
        handlers.append(rich_handler)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [bold green]{%(name)s}[/] - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=handlers,
    )


def save_config_yaml(cfg: DictConfig, output_dir: str, filename: str = "config.yaml") -> str:
    """Save the resolved full config into output_dir/filename."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    OmegaConf.save(config=cfg, f=out_path, resolve=True)
    return out_path
