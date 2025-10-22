"""
Final launcher for the Vehicle Insurance Streamlit Dashboard
------------------------------------------------------------
This file serves as the single entry point for the full Streamlit app.

To launch the dashboard, run:

    streamlit run src/main_py.py
"""

import sys
from pathlib import Path

# ======================================================================
#                    CONFIGURE PYTHON IMPORT PATHS
# ======================================================================

# Get absolute path of this file
CURRENT_FILE = Path(__file__).resolve()

# "src" directory
SRC_DIR = CURRENT_FILE.parent

# "src/functions" directory (where all analysis modules are)
FUNCTIONS_DIR = SRC_DIR / "functions"

# Add both to sys.path if not already present
for path in (SRC_DIR, FUNCTIONS_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

# ======================================================================
#                    IMPORT THE MAIN DASHBOARD FUNCTION
# ======================================================================

try:
    from functions.main_dashboard import main as run_dashboard  # type: ignore
except ModuleNotFoundError as e:
    raise ImportError(
        f"❌ Unable to import 'main_dashboard' from 'functions'.\n"
        f"Ensure that 'src/functions/main_dashboard.py' exists.\n"
        f"Original error: {e}"
    )

# ======================================================================
#                          LAUNCH STREAMLIT
# ======================================================================

def main() -> None:
    """Entry point that launches the full Streamlit dashboard."""
    run_dashboard()


# ======================================================================
#                             EXECUTION
# ======================================================================

if __name__ == "__main__":
    main()

