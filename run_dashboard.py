#!/usr/bin/env python3
"""
Crypto Trading Dashboard Launcher

Launch the web-based trading dashboard for real-time market monitoring,
strategy analysis, and backtesting.

Usage:
    python run_dashboard.py [--port PORT] [--debug]
"""

import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.dashboard.app import CryptoDashboard

def main():
    """Main entry point for dashboard launcher"""
    parser = argparse.ArgumentParser(
        description="Launch Crypto Trading Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_dashboard.py                    # Default: localhost:8050, debug mode
    python run_dashboard.py --port 8080      # Custom port
    python run_dashboard.py --no-debug       # Production mode
        """
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=8050,
        help='Port to run dashboard on (default: 8050)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        default=True,
        help='Run in debug mode (default: True)'
    )
    
    parser.add_argument(
        '--no-debug',
        action='store_true',
        help='Disable debug mode'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    
    args = parser.parse_args()
    
    # Handle debug flag
    debug_mode = args.debug and not args.no_debug
    
    print("=" * 60)
    print("🚀 Crypto Trading Dashboard")
    print("=" * 60)
    print(f"📊 Starting dashboard on http://{args.host}:{args.port}")
    print(f"🔧 Debug mode: {'ON' if debug_mode else 'OFF'}")
    print(f"💡 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Create and run dashboard
        dashboard = CryptoDashboard()
        dashboard.run(debug=debug_mode, port=args.port)
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()