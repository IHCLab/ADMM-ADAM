function [rmse_total, ergas, sam, uiqi, q_band, mssim,psnr] = quality_assessment(ground_truth, estimated, ignore_edges, ratio_ergas)
%quality_assessment - Computes a number of quality indices from the remote sensing literature,
%    namely the RMSE, ERGAS, SAM and UIQI indices. UIQI is computed using 
%    the code by Wang from "A Universal Image Quality Index" (Wang, Bovik).
% 
% ground_truth - the original image (3D image), 
% estimated - the estimated image, 
% ignore_edges - when using circular convolution (FFTs), the borders will 
% probably be wrong, thus we ignore them, 
% ratio_ergas - parameter required to compute ERGAS = h/l, where h - linear
% spatial resolution of pixels of the high resolution image, l - linear
% spatial resolution of pixels of the low resolution image (e.g., 1/4)
% 
%   For more details on this, see Section V.B. of
% 
%   [1] M. Simoes, J. Bioucas-Dias, L. Almeida, and J. Chanussot, 
%        �A convex formulation for hyperspectral image superresolution 
%        via subspace-based regularization,?IEEE Trans. Geosci. Remote 
%        Sens., to be publised.

% % % % % % % % % % % % % 
% 
% Author: Miguel Simoes
% 
% Version: 1
% 
% Can be obtained online from: https://github.com/alfaiate/HySure
% 
% % % % % % % % % % % % % 
%  
% Copyright (C) 2015 Miguel Simoes
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, version 3 of the License.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

% Ignore borders
y = ground_truth(ignore_edges+1:end-ignore_edges, ignore_edges+1:end-ignore_edges, :);
x = estimated(ignore_edges+1:end-ignore_edges, ignore_edges+1:end-ignore_edges, :);

% Size, bands, samples 
sz_x = size(x);
n_bands = sz_x(3);
n_samples = sz_x(1)*sz_x(2);

% RMSE
aux = sum(sum((x - y).^2, 1), 2)/n_samples;
rmse_per_band = sqrt(aux);
rmse_total = sqrt(sum(aux, 3)/n_bands);

% ERGAS
mean_y = sum(sum(y, 1), 2)/n_samples;
ergas = 100*ratio_ergas*sqrt(sum((rmse_per_band ./ mean_y).^2)/n_bands);

% SAM

num = sum(x .* y, 3);
den = sqrt(sum(x.^2, 3) .* sum(y.^2, 3));
index=find(den==0);
den(index)=[];
num(index)=[];

sam = sum(sum(acosd(num ./ den)))/(n_samples-length(index));

% UIQI - calls the method described in "A Universal Image Quality Index"
% by Zhou Wang and Alan C. Bovik
q_band = zeros(1, n_bands);
for idx1=1:n_bands
%     q_band(idx1)=img_qi(ground_truth(:,:,idx1), estimated(:,:,idx1), 32);
    q_band(idx1)=img_qi(ground_truth(:,:,idx1), estimated(:,:,idx1), 64);
end
uiqi = mean(q_band);


%MSSIM
mssim = MSSIM_3D(ground_truth,estimated);

%PSNR

psnr=PSNR_3D(ground_truth,estimated);

function out_ave = PSNR_3D(X3D_ref,X3D_rec)

%Input : (1) X3D_ref: 3D clean HSI data
%        (2) X3D_rec: 3D reconstructed HSI data computed by algorithm
%Output: (1) out_ave: mean psnr of each bands

[~,~,bands]=size(X3D_rec);
X3D_ref=reshape(X3D_ref,[],bands);
X3D_rec=reshape(X3D_rec,[],bands);
msr=mean((X3D_ref-X3D_rec).^2,1);
max2=max(X3D_rec,[],1).^2;
out_ave=mean(10*log10(max2./msr));

%% MSSIM_3D
function k = MSSIM_3D(Y3D_ref, Y3D_rec) 

%Input : (1) Y3D_ref: 3D clean HSI data
%        (2) Y3D_rec: 3D reconstructed HSI data computed by algorithm
%Output: (1) k: mean ssim of each bands

Y3D_rec = (Y3D_rec-min(Y3D_rec(:)))/(max(Y3D_rec(:))-min(Y3D_rec(:)));
Y3D_ref = (Y3D_ref-min(Y3D_ref(:)))/(max(Y3D_ref(:))-min(Y3D_ref(:)));

[row, col, bands] = size(Y3D_ref);%row, column, band
K = [0.01 0.03];
window = fspecial('gaussian', 11, 1.5);

Y2D_rec=reshape(Y3D_rec,[],bands)';
Y2D_ref=reshape(Y3D_ref,[],bands)';
% input size : Bands*observation
L=max(Y2D_ref(:));
for i=1:bands 
    k_tmp(i)  = ssim_index(reshape(Y2D_ref(i,:),[row, col]), reshape(Y2D_rec(i,:),[row, col]), K, window, L);
end
k=mean(k_tmp);