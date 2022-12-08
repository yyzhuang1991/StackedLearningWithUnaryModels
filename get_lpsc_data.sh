 mkdir corpus-LPSC
 cd corpus-LPSC
 wget -O lpsc-annotated.zip https://zenodo.org/record/1048419/files/lpsc-annotated.zip?download=1
 wget -O mer-a.zip https://zenodo.org/record/7066107/files/mer-a.zip?download=1
 wget -O mpf.zip https://zenodo.org/record/7066107/files/mpf.zip?download=1
 wget -O phx.zip https://zenodo.org/record/7066107/files/phx.zip?download=1
 unzip \*.zip
 mv lpsc-annotated/README.txt README-lpsc.txt
 mv lpsc-annotated/* .
 rm -r lpsc-annotated
 rm *.zip 
 cd .. 

# We skip some files in PHX and MPF. A file is skipped if it satisfies one of the following 2 conditions: 
#     1. the .ann file has no Target entities 
#     2. the .ann file has no Component entities and no Property entities.  
cd scripts/
python filter_files.py 
cd .. 