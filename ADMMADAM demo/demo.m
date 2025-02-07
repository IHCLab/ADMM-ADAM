clear all;close all;clc;
addpath(genpath('function'));addpath(genpath('results'));addpath(genpath('HSI_dataset'));
addpath(genpath('deep-hs-prior-master')); % Please first download HDIP from https://github.com/acecreamu/deep-hs-prior
addpath(genpath('tensor_toolbox-v3.1')); % Please first download the tensor toolbox from https://www.tensortoolbox.org/
addpath(genpath('ndSparse_G4_2021_03_16')); % Please first download ndSparse.m from https://fr.mathworks.com/matlabcentral/fileexchange/29832-n-dimensional-sparse-arrays
%% Algorithm
whichcase ='mode 0: Single data(HDIP)';
switch whichcase
    case'mode 0: Single data(HDIP)'%channels can be adjusted
        % load input data
        load('HSI_dataset/ShellKey')
        X3D_corrupted =X3D_ref .* mask_3D;
        % generate single data deep learning solution
        system('conda activate admmadam & python deep-hs-prior-master/deep-hs-prior-master/inpainting2D.py');%activate with your own env (i.e., admmadam)
        load('results/result_inpainting_2D_it01000_192.mat');
        X3D_DL = double(pred);
        % ADMM
        [X3D_rec,time] = ADMMADAM(X3D_corrupted,mask_3D,X3D_DL);
        plot_result(mask_3D,X3D_ref,X3D_DL,X3D_rec,time,time_dl)
    case'mode 1: Small data(GAN)'%fixed 172 channels
        % load input data
        HSIdata_path = "HSI_dataset/Ottawa";
        load(HSIdata_path)
        maskdata_path = "HSI_dataset/mask_Ottawa";
        load(maskdata_path)
        X3D_corrupted =X3D_ref .* mask_3D;        
        % generate small data deep learning solution
        fileID_HSI = fopen('HSI_dataset/val_test.txt', 'w');
        fprintf(fileID_HSI, HSIdata_path);
        fclose(fileID_HSI);
        fileID_mask = fopen('HSI_dataset/val_mask.txt', 'w');
        fprintf(fileID_mask, maskdata_path);
        fclose(fileID_mask);
        system('conda activate admmadam & python function/test.py');
        load('results/gan.mat');
        X3D_DL = double(gan);
        % denormalize
        maxv = max(X3D_corrupted(:));
        minv = min(X3D_corrupted(:));
        X3D_DL_DN = X3D_DL .* (maxv-minv) + minv;
        % ADMM
        [X3D_rec,time] = ADMMADAM(X3D_corrupted,mask_3D,X3D_DL_DN);
        plot_result_2(X3D_ref,X3D_rec,X3D_DL_DN,mask_3D,time,time_dl)
end