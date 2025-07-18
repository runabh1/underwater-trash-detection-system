#!/usr/bin/env python3
"""
Underwater Trash Detection System - Startup Script
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

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'flask', 'opencv-python', 'torch', 'torchvision', 
        'Pillow', 'numpy', 'ultralytics', 'flask-cors'
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
        return install_requirements()
    
    return True

def main():
    """Main startup function"""
    print("🌊 Underwater Trash Detection System")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check model file
    if not check_model_file():
        sys.exit(1)
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    print("\n🚀 Starting the application...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the Flask application
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 