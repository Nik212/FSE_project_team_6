import os


def main():
	"""
	Script for testing based on pretrained models 

	VARIABLES:
		train_path (str) - path to the train.py file of initial repository 
		gpu_num (int) - number of gpu
		train_list (str) - path to list of train files
		data_path_2d (str) - path to 2d image data
		weigth_path (str) - path to txt file of train histogram
		nearest_images (str) - number of nearest images
		model_path (str) - path to pretrained 2d model

	"""

	test_path = "../3dmv/test.py"
	gpu_num = 0
	scene_list = "../data/test/scenes.txt"
	data_path_2d = "../data/test/"
	data_path_3d = "../data/test/"
	nearest_images = 5
	model_path = "../data/models/models/scannetv2/scannet5_model2dfixed.pth"
	model2d_orig_path = "../data/models/models/scannetv2/2d_scannet.pth "

	command = """python3 {train_path} --gpu {gpu_num} --scene_list {train_list}  \n 
			  --model_path {model_path} --data_path_2d {path_2d} \n 
			  --data_path_3d {} model2d_orig_path {} 
			  """
	command_input = ' '.join([line for line in command.split()]) 

	os.system(command_input.format(test_path=test_path, gpu_num=gpu_num,
			  scene_list=scene_list, model_path=model_path, model2d_orig_path=model2d_orig_path,
			  data_path_2d=data_path_2d, data_path_3d=data_path_3d))

if __name__ == "__main__":
	main()
