#!/bin/bash

# Delete individual files
rm -f RottenCore/src/exporter.py
rm -f test_project.rc

# Delete directories
rm -rf attic
rm -rf bitlzss
rm -rf common
rm -rf comp
rm -rf comp2
rm -rf hardware
rm -rf ml
rm -rf old_playback_artifacts
rm -rf playback
rm -rf song
rm -rf streameditor
rm -rf streamrecomp
rm -rf vpxtest

echo "Cleanup completed."
