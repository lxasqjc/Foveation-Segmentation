clear all; close all; clc
list_fileID = fopen('./example_odgt_list.txt','w');
formatSpec = '{"fpath_img": "land-train/%6s", "fpath_segm": "land_train_gt_processed/%6s", "width": 2448, "height": 2448}\n';

path_Cropped_training_images = '/SAN/medic/Histo_MRI_GPU/chenjin/Data/DeepGlobe/land-train';
path_Cropped_gt_images = '/SAN/medic/Histo_MRI_GPU/chenjin/Data/DeepGlobe/land_train_gt_processed';

train_filepaths = dir(fullfile(path_Cropped_training_images, '*_sat.jpg'));
gt_filepaths = dir(fullfile(path_Cropped_gt_images, '*_label.png'));

fprintf('%5d images read\n', length(train_filepaths));
rand_idx = randperm(length(train_filepaths));


for idx = 1:length(rand_idx)
	name_train_im = train_filepaths(rand_idx(idx)).name;
	train_name_base = strsplit(name_train_im,'_sat.');
    name_gt_im = strcat(train_name_base{1}, '_label.png')
	if isfile(fullfile(path_Cropped_gt_images, name_gt_im))
	    fprintf(list_fileID,formatSpec,name_train_im,name_gt_im);
	    fprintf(formatSpec,name_train_im,name_gt_im);
    end
end

fprintf('done\n');
fclose(list_fileID);
