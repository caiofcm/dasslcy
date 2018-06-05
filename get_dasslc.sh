wget http://www.enq.ufrgs.br/enqlib/numeric/dasslc_v39.zip
unzip dasslc_v39.zip -d dasslc
find ./dasslc -type f -not -name 'dasslc.h' -not -name 'dasslc.c' -print0 | xargs -0 rm --
