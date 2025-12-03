#!/usr/bin/env python3
"""
CV Master - Computer Vision Application
Main entry point
"""
import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
import qdarktheme

from ui import MainWindow


def main():
    """Main application entry point"""
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("CV Master")
    app.setOrganizationName("CVMaster")
    app.setApplicationVersion("1.0.0")
    
    # Apply dark theme
    qdarktheme.setup_theme(
        theme="dark",
        custom_colors={
            "[dark]": {
                "primary": "#0078d4",
                "background": "#1e1e1e",
                "border": "#3d3d3d",
            }
        }
    )
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
