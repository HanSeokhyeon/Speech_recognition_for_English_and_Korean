#!/usr/bin/env bash
if [ "$#" -ne 1 ]; then
    echo "Usage : ./timit_preprocess.sh <timit folder>"
fi
echo 'Transfering raw TIMIT wave file format from NIST to RIFF.'
echo ' '
# MFCC
python3 timit_preprocess_mel40_spikegram8_0.py $1 timit_mel_spikegram_144