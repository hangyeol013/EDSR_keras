clear;close all;
%% settings
%% ÇÒ¶§¸¶´Ù folder À§Ä¡ test¶û train ¹Ù²ãÁà¾ß ÇÑ´Ù.

folder = 'Train\DIV2K_train_HR_aug';
%folder = 'Train\DIV2K_validation_HR';

savepath_train = 'train_DIV2Kaug_scale4.h5';
savepath_test = 'cross_val_DIV2K_scale3.h5';

scale = 4;
size_input = 48;
size_label = size_input * scale;


%% initialization
data = zeros(size_input, size_input, 3, 1);
label = zeros(size_label, size_label, 3, 1);


%% generate data

filepaths = dir(fullfile(folder,'*.bmp'));

count = 0;
    
for i = 1 : length(filepaths)
    
    image = imread(fullfile(folder,filepaths(i).name));
    image = im2double(image);

    im_label = modcrop(image, scale);
    im_input = imresize(im_label,1/scale,'bicubic');
    [hei, wid, rgb] = size(im_input);
    
    if hei > 48 && wid > 48
        
        for kk = 1:2

            x_input = randi([1, wid - size_input + 1]);
            y_input = randi([1, hei - size_input + 1]);
           
            subim_input = im_input(y_input : y_input + size_input - 1, x_input : x_input + size_input - 1, :);
            subim_label = im_label((y_input-1)*scale + 1 : (y_input-1)*scale + size_label, (x_input-1)*scale + 1 : (x_input-1)*scale + size_label, :);

            count = count + 1;

            data(:, :, :, count) = subim_input;
            label(:, :, :, count) = subim_label;

        end
            
    end
end

data_dims = size(data);
label_dims = size(label);

%% writing to HDF5

h5create(savepath_train,'/data',data_dims);
h5create(savepath_train, '/label', label_dims);

h5write(savepath_train, '/data', data);
h5write(savepath_train, '/label', label);

h5disp(savepath_train);