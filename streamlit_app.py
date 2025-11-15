#!/usr/bin/env python3
"""
PROMIS Dashboard - Streamlit App Entry Point

This is the entry point for Streamlit Cloud deployment.
It uses the standalone interactive_metrics_viz.py in the same directory
"""

# Import the main function from the standalone version
from interactive_metrics_viz import main

if __name__ == "__main__":
    main()

