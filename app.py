"""Root entry point for Streamlit Cloud deployment."""
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the actual app
from frontend.app import main
main()
