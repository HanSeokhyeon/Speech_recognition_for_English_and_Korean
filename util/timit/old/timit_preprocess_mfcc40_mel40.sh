if [ "$#" -ne 1 ]; then
    echo "Usage : ./timit_preprocess.sh <timit folder>"
fi
echo 'Transfering raw TIMIT wave file format from NIST to RIFF.'
echo ' '
# MFCC
python3 timit_preprocess_mfcc40_mel40.py $1 timit_mfcc_mel_240