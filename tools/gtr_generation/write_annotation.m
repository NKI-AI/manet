clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;  % Erase all existing variables. Or clearvars if you want.
workspace;  % Make sure the workspace panel is showing.
format long g;
format compact;
fontSize = 5;
%spacing=0.07;
%patch_size=8;
%bbox_size=3;
patch_size_mm=0.56;
bbox_size_mm=0.21;
addpath 'C:\Users\bened\Desktop\UNIVERSITA''\dottorato_ricerca_Nijmegen\groundtruth_generation\code\jsonlab-master\jsonlab-master';

%===============================================================================


%===============================================================================
%Load the image and the relative json file

% mount_name='Y:\projects\benedetta\data\';
% annotation_folder='Y:\projects\benedetta\data\annotations\';
% dir_name='C:\Users\bened\Desktop\UNIVERSITA''\dottorato_ricerca_Nijmegen\groundtruth_generation\code\mathwork\';
% dir_out_name='C:\Users\bened\Desktop\UNIVERSITA''\dottorato_ricerca_Nijmegen\groundtruth_generation\code\mathwork2\binary_mask\';
% list_name='hologic_train_calc_stage1.elst';
% dir_dicom='Y:\projects\benedetta\data\features\';
% out_dir='Y:\projects\benedetta\out_folder2\';

mount_name='Z:\LST\';
annotation_folder='Z:\annotations\';
%dir_name='C:\Users\bened\Desktop\UNIVERSITA''\dottorato_ricerca_Nijmegen\groundtruth_generation\code\mathwork\';
%dir_out_name='C:\Users\bened\Desktop\UNIVERSITA''\dottorato_ricerca_Nijmegen\groundtruth_generation\code\mathwork2\binary_mask\';
%list_name='ge_train_calc_stage1.elst';
list_name='ge_test_normals.elst';
dir_dicom='Z:\features\';
out_dir='Y:\projects\benedetta\ge\groundtruth_test_normals\';



list_file_name=strcat(mount_name,list_name);
fid=fopen(list_file_name);
tlines = cell(0,1);
points=cell(0,1);
len=0;
dicom_read=0;
tline = fgetl(fid);
count=0
%tline = fgetl(fid);
%tline = fgetl(fid);
% tline = fgetl(fid);
% tline = fgetl(fid);
% tline = fgetl(fid);
% tline = fgetl(fid);
% tline = fgetl(fid);
% tline = fgetl(fid);
% tline = fgetl(fid);
%%%%fino a qui 
% tline = fgetl(fid);
% tline = fgetl(fid);
% tline = fgetl(fid);
%view={'cl','cr','ml','mr'};
view={'cl','cr','ml','mr'}
skipped_files=0;
dicom_to_load='';
total_max =cell(0,1);
total_min=cell(0,1);
total_mean=cell(0,1);
total_max =cell(0,1);

while ischar(tline)
     tlines{end+1,1} = tline;
     
     if tline==-1
         break
     end  
%      if dicom_read==5
%          break
%      end 
     
     for i=1:4
       suffix= view(i);
       file_name_without_suffix=char(strcat(char(tline),suffix));
       file_name=strcat(file_name_without_suffix,'.json');
       file_name2=strcat(file_name_without_suffix,'.json');
       file_name=strcat(annotation_folder,file_name);

       %disp(file_name);
       if exist(file_name, 'file') == 0
       % File does not exist
       % Skip to bottom of loop and continue with the loop
       skipped_files=skipped_files+1;
       continue;
       else 
        disp(file_name);   
        dat = loadjson(file_name,'SimplifyCell',1);
        output_file_name=strcat('Y:\projects\benedetta\ge\all_annotations\',file_name2)
        savejson(output_file_name,dat)  
         
        if dat.has_annotations==1
            disp(file_name);
            disp(dat.has_annotations);
            
            if isfield(dat.annotations,'points_0')==1
                
                %disp(dat.annotations);
                dicom_repeat=strcat('dp',file_name_without_suffix);
                dicom_repeat=strcat(dicom_repeat,'\');
                dicom_name=strcat(dir_dicom,dicom_repeat);
                %dicom_name=strcat(dicom_name,dicom_repeat);
                dicom_name=strcat(dicom_name,'windowlevel_image.dcm');
                disp(dicom_name);
                [X,MAP]=dicomread(dicom_name);
                dicom_read=dicom_read+1;
                dicomwrite(X,filename)
                
                %figure; imshow(X, 'DisplayRange', []);

                % %Convert the dicom file into an image
                [pathname, name, ext] = fileparts(dicom_name);
                name = strcat(name, '.png');
                %new_name = fullfile(pathname, name);

                image = uint8(255*mat2gray(X));
                %imwrite(image,new_name,'png','Bitdepth',16,'Mode','lossless');
                %png_image = imread (new_name);
                png_image=image;
                %imshow(png_image);
                %title(dicom_name);
                %figure; 
                %disp(dat.id);
                %disp(dat.annotations);
                
                %disp(dat.annotations);
                
                
                %%%%Generate a new image for the mask   
                image_height=size(png_image,1);
                image_width=size(png_image,2);
                mask_image=zeros(image_height,image_width,1,'single');
                
                %%%%%Find point annotation
                points{1,1} = dat.annotations.points_0.points;
                coordinates = cell2mat(points);
                %spacing=dat.annotations.points_0.annotation_spacing(1);
                spacing=dat.annotations.points_0.region_annotation_spacing(1);
                pixel_size=dat.annotation_spacing;
                
               
                patch_size=round(patch_size_mm/spacing);
                bbox_size=round(bbox_size_mm/spacing);
                
                
                
                %[max_numberOFpixels,min_numberOFpixels,mean_numberOFpixels,final_mask_image]=roi_generation5(coordinates,png_image,spacing,mask_image,file_name_without_suffix,out_dir,patch_size,bbox_size,pixel_size,count);
                out_file=strcat(out_dir,file_name_without_suffix);
                out_file1=strcat(out_file,'.png');
                %out_file_fig=strcat(out_file,'_mask.fig');
%                 saveas(gcf,out_file_fig)
                imwrite(final_mask_image,out_file1);
     
%                 figure;
%                 if dicom_read==5
%                  break
%                 end 

            end    
        end
       end
     end 
   
     tline = fgetl(fid);        
     len=len+1;

     

    
     
     
end
disp(dicom_read)
disp(count)
%   final_max=max(cell2mat(total_max));
%   final_min=min(cell2mat(total_min));
%   final_mean=mean(cell2mat(total_mean));
%