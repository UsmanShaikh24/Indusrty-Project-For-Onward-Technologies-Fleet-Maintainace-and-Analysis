#!/usr/bin/env python3
"""
Simple startup script for Fleet Maintenance App
No Docker needed - runs directly with Python!
All ML predictions work locally - no API server required!
"""

import subprocess
import sys
import os

def main():
    print("🚛 FLEET MAINTENANCE PREDICTIVE ANALYTICS")
    print("=" * 60)
    print("Starting the app directly with Python (no Docker, no API needed!)")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists('fleet_maintenance_app.py'):
        print("❌ Error: fleet_maintenance_app.py not found!")
        print("Please run this script from the project root directory.")
        return
    
    # Check if models exist
    if not os.path.exists('models/fleet_maintenance_model.pkl'):
        print("❌ Error: Trained models not found!")
        print("Please run 'python enhanced_ml_engine.py' first to train the models.")
        return
    
    print("✅ All components ready!")
    print()
    print("🚀 Launching Streamlit Dashboard...")
    print("📱 The app will open in your default web browser")
    print("🔗 URL: http://localhost:8501")
    print()
    print("💡 What you get:")
    print("   ✅ Complete ML predictions (no API needed!)")
    print("   ✅ Interactive dashboard with real-time analytics")
    print("   ✅ File upload for batch processing")
    print("   ✅ All features working locally")
    print()
    print("⏹️  Press Ctrl+C to stop the app")
    print("=" * 60)
    
    try:
        # Launch Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "fleet_maintenance_app.py",
            "--server.headless", "false",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 App stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting app: {e}")

if __name__ == "__main__":
    main()
