"""Thin Code Ocean entry point for OASIS fluorescence event detection.

All logic lives in the ``aind-ophys-oasis-event-detection-library`` package;
this wrapper only parses settings (CLI / environment) and invokes ``run``.
"""

from aind_ophys_oasis_event_detection_library.job import run

if __name__ == "__main__":
    run()
