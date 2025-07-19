#!/usr/bin/env python3
"""
Keep Alive Script for Underwater Trash Detection System
This script can be used to ping your Render app to keep it awake
"""

import requests
import time
import sys

def ping_app(url):
    """Ping the app to keep it awake"""
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            print(f"✅ App is awake: {response.status_code}")
            return True
        else:
            print(f"⚠️  App responded with status: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Error pinging app: {e}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python keep_alive.py <your-render-url>")
        print("Example: python keep_alive.py https://your-app.onrender.com")
        sys.exit(1)
    
    url = sys.argv[1]
    if not url.startswith('http'):
        url = 'https://' + url
    
    print(f"🌊 Keeping Underwater Trash Detection System awake...")
    print(f"📡 Pinging: {url}")
    print("⏹️  Press Ctrl+C to stop")
    
    try:
        while True:
            ping_app(url)
            print("💤 Waiting 15 minutes before next ping...")
            time.sleep(900)  # 15 minutes
    except KeyboardInterrupt:
        print("\n👋 Stopping keep-alive script")

if __name__ == '__main__':
    main() 