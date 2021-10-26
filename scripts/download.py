from pathlib import Path
from shutil import rmtree
import os
import platform
import zipfile
import requests 
from clint.textui import progress


def download_file(url_path, location_path, block_size):
	"""
	Downloads an archive from the $url_path to $location_path.

	VARIABLES:
		url_path (str) - url of .zip an archive 
		location_path (str) - location, where a dataset will be stored  
		block_size (int) - size of block that is downloaded per iteration
	"""

	Path(location_path).mkdir(parents=True)
	rqsts = requests.get(url_path, stream=True)
	file_downloaded = location_path + "/file.zip"

	with open(file_downloaded, 'wb') as file:
		total_length = int(rqsts.headers.get('content-length'))
		for chunk in progress.bar(rqsts.iter_content(chunk_size=block_size), 
                                          expected_size=(total_length/block_size) + 1): 
			if chunk:
				file.write(chunk)
				file.flush()

	# extracting data from the archive 			
	with zipfile.ZipFile(file_downloaded, 'r') as file_zip:
		file_zip.extractall(location_path)
	os.remove(file_downloaded) # removing the archive 


def make_dataset(url_path, location_path, block_size):
	"""
	Does all processing with downloaded archive, including unzipping and removing archive. 

	VARIABLES:
		url_path (str) - url of .zip an archive 
		location_path (str) - location, where a dataset will be stored  
		block_size (int) - size of block that is downloaded per iteration
	"""

	if not Path(location_path).is_dir():
		download_file(url_path=url_path, location_path=location_path, block_size=block_size)	
	else:
		print("Testing dataset already exists, check the directory: '{}'".format(location_path))
		user_mark = True
		while user_mark:
			response = input("Do you want to rewrite the directory? [y/n] ")
			if response == "y":
				rmtree(location_path)
				user_mark = False
				download_file(url_path=url_path, location_path=location_path, block_size=block_size)	
			elif response == "n":
				user_mark = False
			else:
				pass


def main():
	"""
	Determines the OS and execute all downloading and processing processes for archives.
	"""

	url_train = "http://kaldir.vc.in.tum.de/adai/3DMV/data/3dmv_scannet_v2_train.zip"
	url_test = "http://kaldir.vc.in.tum.de/adai/3DMV/data/3dmv_scannet_v2_test_scenes.zip"
	block_size = 1024	# block size in bytes  
	system = platform.system()
	train_path = "../data/train"
	test_path = "../data/test"

	if system == "Windows":
		train_path = os.path.abspath(train_path)
		test_path = os.path.abspath(test_path)
	elif system == "Linux":
		pass
	else:
		raise Exception("Only Windows or Linux systems are supported")
	
	# creating a directory for the data 
	Path("../data").mkdir(parents=True, exist_ok=True)

	# downloading data
	make_dataset(url_path=url_train, location_path=train_path, block_size=block_size)
	make_dataset(url_path=url_test, location_path=test_path, block_size=block_size)


if __name__ == "__main__":
	main()
