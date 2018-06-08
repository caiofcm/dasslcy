from __future__ import print_function
import os
import sys
import zipfile
# import urllib

DASSLC_FOLDER = "dasslc_base"
DASSLC_ZIP_FILE = "dasslc_v39.zip"

def get_dasslc():
	url = "http://www.enq.ufrgs.br/enqlib/numeric/dasslc_v39.zip"
	out_file = DASSLC_ZIP_FILE
	dir_dasslc = DASSLC_FOLDER
	if not os.path.isdir("./dasslc_base"):
		print('DOWNLOADING dasslc files')
		download(url, out_file)
		unzip_file(out_file, dir_dasslc)
		remove_files_except()
		os.remove(out_file)
	else:
		print('dasslc_base folder exists: NOT DOWNLOADING')

def download(url, out_file):
	if sys.version_info >= (3, 0):
		import urllib.request
		urllib.request.urlretrieve(url, out_file)
	else:
		import urllib
		urllib.urlretrieve(url, out_file)  # pylint: disable=maybe-no-member

def unzip_file(path_, out_dir):
	zip_ref = zipfile.ZipFile(path_, 'r')
	zip_ref.extractall(out_dir)
	zip_ref.close()

def get_file_names(dir):
	return [name for name in os.listdir(
		dir) if os.path.isfile(os.path.join(dir, name))]

def remove_files_except():
	files_names = get_file_names(DASSLC_FOLDER)
	for fn in files_names:
		if not (fn == "dasslc.h" or fn == "dasslc.c"):
			fn_rel = DASSLC_FOLDER + "/" + fn
			os.remove(fn_rel)
	return

def main():
	get_dasslc()

if __name__ == '__main__':
	main()
