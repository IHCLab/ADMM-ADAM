function plot_result(mask_3D,X3D_ref,X3D_DL,X3D_rec,time,time_dl)

time_all=time+time_dl;
corrupted_bands = [1:172];
[ ~ , ergas_DL , sam_DL , uiqi_DL , ~ , mssim_DL , psnr_DL ] = quality_assessment(X3D_ref(:,:,corrupted_bands),X3D_DL(:,:,corrupted_bands), 0, 1/6);
[ ~ , ergas_rec, sam_rec, uiqi_rec, ~ , mssim_rec, psnr_rec] = quality_assessment(X3D_ref(:,:,corrupted_bands),X3D_rec(:,:,corrupted_bands)  , 0, 1/6);

figure()
subplot(1,4,1)
band_set=[25 15 6]; % RGB bands
normColor=@(R)max(min((R-mean(R(:)))/std(R(:)),2),-2)/3+0.5;
temp_show=X3D_ref(:,:,band_set);
temp_show=normColor(temp_show);
imshow(temp_show);title({'Reference';''});xlabel(["PSNR:","UIQI:","ERGAS:","SAM:","SSIM:","TIME:"]);

subplot(1,4,2)
temp_show=temp_show.*mask_3D(:,:,25).*mask_3D(:,:,15).*mask_3D(:,:,6);
imshow(temp_show);title({'Corruption';''});
redline = [22:23 50 70 100:102 150:151 180:181];
for ii = 1:length(redline)
hold on;
line([redline(ii),redline(ii)],[1,256],'Color',[0.5,0,0],'LineWidth',1)
end


subplot(1,4,3)
band_set=[25 15 6]; % RGB bands
normColor=@(R)max(min((R-mean(R(:)))/std(R(:)),2),-2)/3+0.5;
temp_show=X3D_DL(:,:,band_set);
temp_show=normColor(temp_show);
imshow(temp_show);title({['HDIP:    '];['iter = ' ,num2str(1000)]});xlabel([round(psnr_DL,3), round(uiqi_DL,3), round(ergas_DL,3),round( sam_DL,3), round(mssim_DL,3), round(time_dl,3)]);


subplot(1,4,4)
band_set=[25 15 6]; % RGB bands
normColor=@(R)max(min((R-mean(R(:)))/std(R(:)),2),-2)/3+0.5;
temp_show=X3D_rec(:,:,band_set);
temp_show=normColor(temp_show);
imshow(temp_show);title({['ADMM-HDIP: '];['iter = ', num2str(1000)]});xlabel([round(psnr_rec,3), round(uiqi_rec,3),round( ergas_rec,3), round(sam_rec,3),round( mssim_rec,3),round(time_all,3)]);