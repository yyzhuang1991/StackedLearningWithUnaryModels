
cd scripts/

STANFORD_DIR=$1
python parse_texts.py \
        --indir ../corpus-LPSC/lpsc15 \
                ../corpus-LPSC/lpsc16 \
                ../corpus-LPSC/mpf \
                ../corpus-LPSC/phx \
                ../corpus-LPSC/mer-a \
        --outdir ../parse \
        --corenlpDir $STANFORD_DIR 
cd .. 