# |---------------------------------------------------------|
# |                                                         |
# |                 Give Feedback / Get Help                |
# | https://github.com/getbindu/Bindu/issues/new/choose    |
# |                                                         |
# |---------------------------------------------------------|
#
#  Thank you users! We ❤️ you! - 🌻

"""investor-agent - An Bindu Agent."""

from investor_agent.__version__ import __version__
from investor_agent.main import (
    handler,
    initialize_agent,
    initialize_all,
    main,
    run_agent,
)

__all__ = [
    "__version__",
    "handler",
    "initialize_agent",
    "initialize_all",
    "main",
    "run_agent",
]
