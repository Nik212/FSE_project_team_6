import os


def main():
	"""
	Script for training based on pretrained models 

	VARIABLES:
		train_path (str) - path to the train.py file of initial repository 
		gpu_num (int) - number of gpu
		train_list (str) - path to list of train files
		data_path_2d (str) - path to 2d image data
		weigth_path (str) - path to txt file of train histogram
		nearest_images (str) - number of nearest images
		model_path (str) - path to pretrained 2d model

	"""

	train_path = "../3dmv/train.py"
	gpu_num = 0
	train_list = "../data/train/"
	data_path_2d = "../data/train/"
	weigth_path = "../data/train/counts.txt"
	nearest_images = 5
	model_path = "../data/models/models/scannetv2/scannet5_model2dfixed.pth"

	command = """python3 {train_path} --gpu {gpu_num} --train_data_list {train_list}  \n 
			  --data_path_2d {path_2d} --class_weight_file {weight_file} \n 
			  --num_nearest_images {nearest_images} --model2d_path {model_path}
			  """
	command_input = ' '.join([line for line in command.split()]) 

	os.system(command_input.format(train_path=train_path, gpu_num=gpu_num,
			  train_list=train_list, path_2d=data_path_2d, weight_file=weigth_path, 
			  nearest_images=nearest_images, model_path=model_path))

if __name__ == "__main__":
	main()
