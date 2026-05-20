from __future__ import annotations

import sys
from importlib import resources


def main() -> None:
    """Launch the packaged Streamlit UI."""
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError as exc:
        raise SystemExit(
            "The Streamlit UI dependencies are not installed. "
            "Install them with `pip install episcope[ui]` or `pip install .[ui]`."
        ) from exc

    app_path = resources.files("episcope.ui").joinpath("streamlit_app.py")
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        *sys.argv[1:],
    ]
    streamlit_cli.main()
