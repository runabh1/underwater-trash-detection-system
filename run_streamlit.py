#!/usr/bin/env python3
"""
Streamlit Underwater Trash Detection System - Startup Script
"""

import sys
import os
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_model_file():
    """Check if the model file exists"""
    if not os.path.exists('best.pt'):
        print("❌ Error: Model file 'best.pt' not found")
        print("Please ensure your trained model file is in the current directory")
        return False
    print("✅ Model file 'best.pt' found")
    return True

def install_streamlit_requirements():
    """Install Streamlit requirements"""
    print("📦 Installing Streamlit requirements...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements_streamlit.txt'])
        print("✅ Streamlit requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def check_streamlit_dependencies():
    """Check if Streamlit and required packages are installed"""
    required_packages = [
        'streamlit', 'opencv-python-headless', 'torch', 'torchvision', 
        'Pillow', 'numpy', 'ultralytics'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        return install_streamlit_requirements()
    
    return True

def main():
    """Main startup function"""
    print("🌊 Streamlit Underwater Trash Detection System")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check model file
    if not check_model_file():
        sys.exit(1)
    
    # Check dependencies
    print("\n🔍 Checking Streamlit dependencies...")
    if not check_streamlit_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    print("\n🚀 Starting Streamlit application...")
    print("📱 Open your browser and go to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Start Streamlit
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py', '--server.port=8501'])
    except KeyboardInterrupt:
        print("\n👋 Streamlit application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting Streamlit: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 