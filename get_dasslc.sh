if [ ! -d "./dasslc_base" ]; then
echo "DOWNLOADING FILES"
    wget http://www.enq.ufrgs.br/enqlib/numeric/dasslc_v39.zip
    unzip -o dasslc_v39.zip -d dasslc_base
    find ./dasslc_base -type f -not -name 'dasslc.h' -not -name 'dasslc.c' -print0 | xargs -0 rm --
    rm -f dasslc_v39.zip
else
    echo "PATH dasslc_base EXISTS, not downloading files"
fi
