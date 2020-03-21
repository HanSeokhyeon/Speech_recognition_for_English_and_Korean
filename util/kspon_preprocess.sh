if [ "$#" -ne 1 ]; then
    echo "Usage : ./kspon_preprocess.sh <kspon folder>"
fi
echo ' '
# MFCC
python3 kspon_preprocess.py $1 kspon_mfcc_39