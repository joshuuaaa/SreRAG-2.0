#!/usr/bin/env python3
"""
Audio debugging tool for crisis-assistant.
Helps diagnose microphone and Whisper STT issues.
"""

import sys
import os

def main():
    print("🎤 Crisis Assistant Audio Debug")
    print("=" * 50)
    
    # Check imports
    try:
        import sounddevice as sd
        import soundfile as sf
        import numpy as np
        print("✅ Audio libraries imported successfully")
    except ImportError as e:
        print(f"❌ Missing audio library: {e}")
        print("Install with: pip install sounddevice soundfile numpy")
        return 1
    
    # List available devices
    print("\n📱 Available audio devices:")
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                mark = "*" if i == sd.default.device[0] else " "
                print(f"  {mark} {i}: {device['name']} ({device['max_input_channels']} channels, {device['default_samplerate']} Hz)")
    except Exception as e:
        print(f"❌ Error querying devices: {e}")
        return 1
    
    # Test default device recording
    print(f"\n🔴 Testing default microphone (device {sd.default.device[0]})...")
    print("Recording 2 seconds...")
    
    try:
        sr = 16000
        duration = 2
        audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        
        audio = np.squeeze(audio)
        rms = float(np.sqrt(np.mean(audio**2)))
        
        print(f"  RMS level: {rms:.6f}")
        print(f"  Min/Max: {float(audio.min()):.6f} / {float(audio.max()):.6f}")
        
        if rms < 1e-6:
            print("❌ SILENT - No audio captured!")
            print("\nTroubleshooting:")
            print("1. Check if microphone is muted")
            print("2. Try: sudo usermod -aG audio $USER")
            print("3. Log out and back in after adding to audio group") 
            print("4. Check PipeWire/PulseAudio settings")
            print("5. Test with: arecord -D default -f S16_LE -r 16000 -c 1 -d 2 test.wav")
        else:
            print("✅ Audio captured successfully!")
            
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        return 1
    
    # Check Whisper binary
    print("\n🤖 Checking Whisper STT...")
    
    try:
        from src.audio.stt import STTConfig
        import yaml
        
        cfg = yaml.safe_load(open('configs/config.yaml'))
        stt_cfg_d = cfg.get('voice', {}).get('stt', {})
        
        whisper_bin = stt_cfg_d.get('whisper_bin', 'whisper-cli')
        model_path = stt_cfg_d.get('model_path', 'models/whisper/ggml-tiny.en.bin')
        
        print(f"  Binary: {whisper_bin}")
        print(f"  Model: {model_path}")
        
        if os.path.exists(whisper_bin):
            print("✅ Whisper binary exists")
        else:
            print("❌ Whisper binary not found")
            
        if os.path.exists(model_path):
            print("✅ Whisper model exists")
        else:
            print("❌ Whisper model not found")
            
    except Exception as e:
        print(f"❌ Whisper check failed: {e}")
        return 1
    
    print(f"\n🏁 Audio debug complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())