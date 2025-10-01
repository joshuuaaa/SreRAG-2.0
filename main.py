#!/usr/bin/env python3
"""Crisis Assistant - Entry Point"""
import argparse
from src.orchestrator import Orchestrator

def main():
    parser = argparse.ArgumentParser(description="Crisis Assistant - Offline Emergency AI")
    parser.add_argument('--voice', action='store_true', help='Enable voice mode (coming in Phase 2)')
    args = parser.parse_args()
    
    if args.voice:
        print("🎤 Voice mode not yet implemented. Stay tuned for Phase 2!")
        return
    
    # Run text mode
    assistant = Orchestrator()
    assistant.run_text_loop()

if __name__ == "__main__":
    main()
    