clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;  % Erase all existing variables. Or clearvars if you want.
workspace;  % Make sure the workspace panel is showing.
format long g;
format compact;
fontSize = 25;
pixel_size=200;
spacing=0.07;
patch_size=13;
method='watershed';
%===============================================================================
%Load the image and the relative json file

dir_name='C:\Users\bened\Desktop\UNIVERSITA''\dottorato_ricerca_Nijmegen\groundtruth_generation\code\mathwork\';
dir_out_name='C:\Users\bened\Desktop\UNIVERSITA''\dottorato_ricerca_Nijmegen\groundtruth_generation\code\mathwork2\binary_mask\';
%Open the dicom file and display it
filename_dicom='windowlevel_image.dcm';
full_file = strcat(dir_name,filename_dicom);
[X,MAP]=dicomread(fullfile(dir_name, filename_dicom));
%figure; imshow(X, 'DisplayRange', []);

% %Convert the dicom file into an image
[pathname, name, ext] = fileparts(full_file);
name = strcat(name, '.png');
new_name = fullfile(pathname, name);

image = uint16(65535*mat2gray(X));
imwrite(image,new_name,'png','Bitdepth',16,'Mode','lossless');
png_image = imread (new_name);
imshow(png_image);
figure; 

%look for the json file
theFiles = dir('*.*.png');
filename_ext=theFiles.name;
filename_split=strsplit(filename_ext,'.');
filename=filename_split(1);
filename=char(filename);
filename=strcat(filename,'_json');
filename=strcat(filename,'.txt');
filename=strcat(dir_name,filename);

%start reading the file
fid=fopen(filename); 
tline = fgetl(fid);
tlines = cell(0,1);
len=0;
while ischar(tline)
    tlines{end+1,1} = char(tline);
    tline = fgetl(fid);
    len=len+1;
end

%looking for points
X = ~cellfun('isempty',strfind(cellstr(tlines),'points'))
Points=find(X)
StartLine=Points(2)+2;

X2 = ~cellfun('isempty',strfind(cellstr(tlines),'region_id'))
Points2=find(X2)
EndLine=Points2;
fclose(fid)

%skip the first lines
fid=fopen(filename,'rt');
for k=1:StartLine-1
  fgetl(fid); % read and dump
end

%load the lines of interest
len2=0
tline = fgetl(fid);
tlines = cell(0,1);
for k=StartLine:EndLine-1
   %disp(tline)
   %tline = fgetl(fid);
   tlines{end+1,1} = tline;
   tline = fgetl(fid);
   len2=len2+1;     
end
fclose(fid);

 %generate a new image for the mask   
 image_height=size(png_image,1);
 image_width=size(png_image,2);
 mask_image=zeros(image_height,image_width,1,'single');

 %manipulate lines of interest on the json file
 for i=1:4:len2-1
 %for i=9:9
 tlines_char_X=char(tlines(i));
 tlines_char_Y=char(tlines(i+1));
 %disp('coordinate');
 %disp(tlines_char_X);
 %disp(tlines_char_Y);
 X_ph=str2num(tlines_char_X);
 Y_ph=str2num(tlines_char_Y);
 X_pix=round(X_ph*[(pixel_size/1000)/spacing]);
 Y_pix=round(Y_ph*[(pixel_size/1000)/spacing]);
 
 cropped = png_image(Y_pix - patch_size : Y_pix + patch_size, X_pix - patch_size : X_pix + patch_size, :);
%  figure;
%  subplot(1,2,1)
%  imshow(cropped)
%  hold on; 
%  th = 0:2*pi/10:2*pi;
%  xunit = round(2 * cos(th) + X_pix);
%  yunit = round(2 * sin(th) + Y_pix);
%  h = plot(xunit, yunit,'r-', 'LineWidth', 3);
%  hold off;
 %%%%%%% operazioni sull'immagine originale
 
 %DoG3(cropped);
 %binarization=clustering(cropped);
 %apply_watershed(cropped);
 %binarization=apply_watershed_with_markers(cropped);
 %mask_image(Y_pix - patch_size : Y_pix + patch_size, X_pix - patch_size : X_pix + patch_size, :)=binarization;
 %imshow(mask_image);
 imshow(png_image);  
 centx = X_pix;
 centy = Y_pix;
 r = 3;
 hold on;
 theta = 0 : (2 * pi / 100) : (2 * pi);
 pline_x = round(r * cos(theta) + centx);
 pline_y = round(r * sin(theta) + centy);
 k = ishold;
 plot(pline_x, pline_y, 'r', 'LineWidth', 2);
 hold off;
 figure;
 max_value=image(Y_pix,X_pix);
 center_points=find_new_center(png_image,pline_x,pline_y,max_value,X_pix,Y_pix);
 center_points_mat=cell2mat(center_points);
 X_pix_new=center_points_mat(1,1);
 Y_pix_new=center_points_mat(1,2);
 cropped_new = image(Y_pix_new - patch_size : Y_pix_new + patch_size, X_pix_new - patch_size : X_pix_new + patch_size, :);
 binarization=apply_watershed_with_markers_new_and_old(cropped_new,cropped);
 
 %%%triying to figure out how it works on a image-basis level
 end
 
 


 
 
% center = [X_pix Y_pix]; %XY location of circle center
% radius = 3;
% matSize = 50;
% [X,Y] = meshgrid(1:matSize,1:matSize);
% distFromCenter = sqrt((X-center(1)).^2 + (Y-center(2)).^2);
% onPixels = abs(distFromCenter-radius) < 0.5;
% [yPx,xPx] = find(onPixels); % THIS is your answer!
% figure, plot(xPx,yPx,'k', center(1),center(2),'ro')
% rectangle('Position',[center-radius [2 2]*radius], 'Curvature',[1 1])
% axis image 







 
%%%% show the image
%imshow(mask_image);
%CC = bwconncomp(mask_image);
%%%%%save the image
% binary_name=strcat(dir_out_name,method);
% binary_name=strcat(binary_name,int2str((2*patch_size)));
% binary_name=strcat(binary_name,'.png');
% imwrite(mask_image,binary_name,'png');
