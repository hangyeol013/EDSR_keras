%% New
folder = 'Train/HR/DIV2K_train_HR';
savepath = 'Train/HR_aug/DIV2K_train_HR_aug/';

filepaths = dir(fullfile(folder,'*.png'));

for i = 1 : length(filepaths)
    filename = filepaths(i).name;
    [add, im_name, tpye] = fileparts(filepaths(i).name);
    
    image = imread(fullfile(folder, filename));
    imwrite(image, [savepath im_name, '.png'])
    
    image_hflip = flipdim(image,2);
    imwrite(image_hflip, [savepath im_name, '_hflip' '.png']);

    image_vflip = flipdim(image,1);
    imwrite(image_vflip, [savepath im_name, '_vflip' '.png']);

    image_vhflip = flipdim(image_vflip, 2);
    imwrite(image_vhflip, [savepath im_name, '_vhflip' '.png']);
    
    image_trans = permute(image, [2 1 3]);
    imwrite(image_trans, [savepath im_name, '_trans' '.png']);
    
    image_trans_h = permute(image_hflip, [2 1 3]);
    imwrite(image_trans_h, [savepath im_name, '_trans_h' '.png']);
    
    image_trans_v = permute(image_vflip, [2 1 3]);
    imwrite(image_trans_v, [savepath im_name, '_trans_v' '.png']);
    
    image_trans_vh = permute(image_hflip, [2 1 3]);
    imwrite(image_trans_vh, [savepath im_name, '_trans_vh' '.png']);
    

end
