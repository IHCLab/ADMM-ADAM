function plot_result_2(X3D_ref,X3D_reconstruct,X3D_DL,mask_3D,time,time_dl)

% load time_dl 
time_all=time_dl+time;

corrupted_bands = [11:60];
[~, ergas_DL, ~, uiqi_DL, ~,mssim_DL,psnr_DL] = quality_assessment(X3D_ref(:,:,corrupted_bands),X3D_DL(:,:,corrupted_bands), 0, 1/6);
[~, ~, sam_DL, ~, ~] = quality_assessment(X3D_ref(:,:,:),X3D_DL(:,:,:), 0, 1/6);

[~, ergas, ~, uiqi, ~,mssim,psnr] = quality_assessment(X3D_ref(:,:,corrupted_bands),X3D_reconstruct(:,:,corrupted_bands), 0, 1/6);
[~, ~, sam, ~, ~] = quality_assessment(X3D_ref(:,:,:),X3D_reconstruct(:,:,:), 0, 1/6);

figure()
subplot(1,4,1)
FalseColorf=X3D_ref(:,:,[18 8 2]);
RGBmax= max(max(max(X3D_ref)));
RGBmin= min(min(min(X3D_ref)));
FalseColorf= (FalseColorf-RGBmin)/(RGBmax-RGBmin);
xf=imadjust(FalseColorf,stretchlim(FalseColorf),[]);
imshow(xf);title('Reference');xlabel(["PSNR:","UIQI:","ERGAS:","SAM:","SSIM:","TIME:"]);

subplot(1,4,2)
FalseColorf=X3D_ref(:,:,[18 8 2]);
RGBmax= max(max(max(X3D_ref)));
RGBmin= min(min(min(X3D_ref)));
FalseColorf= (FalseColorf-RGBmin)/(RGBmax-RGBmin);
xf=imadjust(FalseColorf,stretchlim(FalseColorf),[]);
xf=xf.*mask_3D(:,:,25).*mask_3D(:,:,15).*mask_3D(:,:,6);
imshow(xf);title('Corruption');

subplot(1,4,3)
FalseColorf=X3D_DL(:,:,[18 8 2]);
RGBmax= max(max(max(X3D_DL)));
RGBmin= min(min(min(X3D_DL)));
FalseColorf= (FalseColorf-RGBmin)/(RGBmax-RGBmin);
xf=imadjust(FalseColorf,stretchlim(FalseColorf),[]);
imshow(xf);title('sdADAM');xlabel([round(psnr_DL,3), round(uiqi_DL,3),round(ergas_DL,3), round(sam_DL,3),round(mssim_DL,3),round(time_dl,3)]);

subplot(1,4,4)
FalseColorf=X3D_reconstruct(:,:,[18 8 2]);
RGBmax= max(max(max(X3D_reconstruct)));
RGBmin= min(min(min(X3D_reconstruct)));
FalseColorf= (FalseColorf-RGBmin)/(RGBmax-RGBmin);
xf=imadjust(FalseColorf,stretchlim(FalseColorf),[]);
imshow(xf);title('ADMM-ADAM');xlabel([round(psnr,3),round(uiqi,3), round(ergas,3), round(sam,3),round(mssim,3),round(time_all,3)]);