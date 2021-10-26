from pathlib import Path
import os
import urllib.request 


def change_print_2_to_3(file_path):
	'''
	Replaces print in format of 2nd versions python to the 3rd one 
	
	VARIABLES:
		file_path (str) - path to the file
	'''

	with open(file_path, 'r', encoding='UTF-8') as file:
		lines = file.readlines()

	for idx, line in enumerate(lines):
		if "print " in line:
			line_old = line.split()
			line_new = '    ' + line_old[0] + '(' + ' '.join(line_old[1:]) + ')' + "\n"
			lines[idx] = line_new

	with open(file_path, 'w', encoding='UTF-8') as file:
		file.writelines(lines)


def download_files():
	'''
	Downloads files, required for the data preparation

	VARIABLES:
		url_utils (str) - url link to the util.py 
		url_sensors (str) - url link to SensorData.py
		utils_path (str) - location where to save util.py
		sensors_path (str) - location where to save SensorData.py
	'''

	#url_utils = "https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/util.py"
	url_utils = "https://raw.githubusercontent.com/ScanNet/ScanNet/master/BenchmarkScripts/util.py"
	url_sensors = "https://raw.githubusercontent.com/ScanNet/ScanNet/master/SensReader/python/SensorData.py"
	utils_path = "../prepare_data/util.py"
	sensors_path = "../prepare_data/SensorData.py"

	# downloading utils.py if it is not installed  
	scannet_file = Path(utils_path)
	if not scannet_file.is_file():
		with urllib.request.urlopen(url_utils) as file:
			content = file.read().decode('utf-8')
		with open(utils_path, 'w', encoding='UTF-8') as file:
			file.write(content)	
		change_print_2_to_3(utils_path)

	else:
		print("util.py is already installed")
	 
	# downloading SensorData.py if it is not installed  
	scannet_file = Path(sensors_path)
	if not scannet_file.is_file():
		with urllib.request.urlopen(url_sensors) as file:
			content = file.read().decode('utf-8')
		with open(sensors_path, 'w', encoding='UTF-8') as file:
			file.write(content)
		change_print_2_to_3(sensors_path)

	else:
		print("SensorData.py is already installed")
		change_print_2_to_3(sensors_path)


def main():
	"""
	VARIABLES:
		prepare_path (str) - path to preparation script
		input_path (str) - path to input files in format that is used in ScanNet
		out_path (str) - path to output files 
		keys (str) - additional keys that are used
	"""

	prepare_path = "../prepare_data/prepare_2d_data.py"
	input_path = "../data/scannetv2"
	out_path = "../data/scannetv2_images"
	keys = "--export_label_images"
	
	# downloading files
	download_files()

	# running preparation in format of ScanNet
	os.system("python {prepare} --scannet_path {input} --output_path {out} {keys}".format(prepare=prepare_path,
				input=input_path, out=out_path, keys=keys) )


if __name__ == "__main__":
	main()
