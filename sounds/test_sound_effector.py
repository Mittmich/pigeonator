#!/usr/bin/env python3
"""
Standalone script to test SoundEffector in isolation.

Usage (from the repo root):
    python sounds/test_sound_effector.py
    python sounds/test_sound_effector.py sounds/crow_1.mp3
"""

import sys
import time
from datetime import timedelta, datetime
from multiprocessing import Pipe

import pygame

# Allow running from anywhere inside the repo
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from birdhub.effectors import SoundEffector

# ---------------------------------------------------------------------------
# Configuration – edit these or pass a sound file as the first argument
# ---------------------------------------------------------------------------
SOUND_FILE   = sys.argv[1] if len(sys.argv) > 1 else "sounds/crow_1.mp3"
TARGET_CLASS = "pigeon"
VOLUME       = 90          # hardware volume percentage (0-100)
# ---------------------------------------------------------------------------

print(f"Initialising SoundEffector with '{SOUND_FILE}' at {VOLUME}% volume …")

effector = SoundEffector(
    target_classes=[TARGET_CLASS],
    cooldown_time=timedelta(seconds=0),   # no cooldown for testing
    config={
        "sound_file": SOUND_FILE,
        "sound_volume": 100,
        # optional overrides:
        # "alsa_card_id":         "2",          # card used for amixer volume control
        # "sdl_audio_driver":     "pulseaudio", # change to "alsa" if no PipeWire/PulseAudio
        # "alsa_device":          "plughw:2,0", # only set when sdl_audio_driver="alsa"
        # "alsa_volume_controls": {"Master": 90, "Channels": 100},  # Channels always 100%
    },
)

# register_detection requires a pipe connection to send events back;
# wire one up manually so the effector can run without a full VideoEventManager.
parent_conn, child_conn = Pipe()
effector._event_manager_connection = parent_conn

# Build a minimal detection dict that satisfies SoundEffector's expectations
detection = {
    "frame_timestamp": datetime.now(),
    "meta_information": {
        "most_likely_object": TARGET_CLASS,
    },
}

print("Triggering playback …")
effector.register_detection([detection])

# Wait for the sound to finish
while pygame.mixer.get_busy():
    time.sleep(0.1)

# Read the event that was sent back through the pipe
if parent_conn.poll():
    event_type, event_data = parent_conn.recv()
    print(f"\nEvent received: {event_type}")
    for key, value in event_data.items():
        print(f"  {key}: {value}")

print("\nDone.")
