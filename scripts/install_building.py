#!/usr/bin/env python3
"""Copy the custom building model into Sinergym's data/buildings folder."""
import os, shutil, sys
import sinergym

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "building", "OfficeMedium_MultiAgent_perfloor.epJSON")
DST_DIR = os.path.join(os.path.dirname(sinergym.__file__), "data", "buildings")

if not os.path.exists(SRC):
    sys.exit(f"missing {os.path.normpath(SRC)} (see building/README.md)")
if not os.path.isdir(DST_DIR):
    sys.exit(f"Sinergym building folder not found: {DST_DIR}")
shutil.copy2(SRC, DST_DIR)
print(f"installed -> {os.path.join(DST_DIR, os.path.basename(SRC))}")
