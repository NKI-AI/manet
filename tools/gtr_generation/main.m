clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;  % Erase all existing variables. Or clearvars if you want.
workspace;  % Make sure the workspace panel is showing.
format long g;
format compact;
fontSize = 25;

%===============================================================================
%Load the image and the relative csv file

dir_name='C:\Users\bened\Desktop\UNIVERSITA''\dottorato_ricerca_Nijmegen\groundtruth_generation\code\mathwork\';

%Open the dicom file
filename_dicom='windowlevel_image.dcm';
full_file = strcat(dir_name,filename_dicom);
Dic_data = dicomread(full_file);
%figure; imshow(Dic_data, 'DisplayRange', []);
%Convert the dicom file into an image
% the name for your image after convertion.
[pathname, name, ext] = fileparts(full_file);
name = strcat(name, '.png');
new_name = fullfile(pathname, name);


% save the image as .jpg format
output_pathname=strcat(dir_name,new_name);
imwrite(Dic_data,new_name,'png','Bitdepth',16,'Mode','lossless');
figure; imshow(new_name, 'DisplayRange', []);
% if isa(Dic_data, 'int16')
%     imwrite(Dic_data,new_name,'jpg','Bitdepth',16,'Mode','lossless');
%     disp('I am here')
% elseif isa(Dic_data, 'uint8')
%     imwrite(Dic_data,new_name,'jpg','Mode','lossless');
%     disp('otherwise')
% end







theFiles = dir('*.*.jpg');
filename_ext=theFiles.name;
filename_split=strsplit(filename_ext,'.');
filename=filename_split(1);
filename=char(filename);
filename=strcat(filename,'.txt');
filename=strcat(dir_name,filename);
%Examples
%Read and display the file fgetl.m one line at a time:
% fid = fopen(filename);
% tline = fgetl(fid);
% while ischar(tline)
%     disp(tline)
%     tline = fgetl(fid);
% end
% fclose(fid);
% 
% %Examples 
% %Read from a specific line
% % open the file
fid=fopen(filename); 
% set linenum to the desired line number that you want to import
linenum = 41;
% use '%s' if you want to read in the entire line or use '%f' if you want to read only the first numeric value
C = textscan(fid,'%s, %f',2,'delimiter',':', 'headerlines',linenum-1);
N_points=C(1,2);
N_points=cell2mat(N_points);
%C=char(C);
%N_points=strsplit(C ,',');

%Example 
%Read from line N to M
fid=fopen(filename,'rt');
StartLine=43;
for k=1:StartLine-1
  fgetl(fid); % read and dump
end
%Fline=fgetl(fid); % this is the 100th line
%do stuff
tline = fgetl(fid);
tlines = cell(0,1);
while ischar(tline)
    disp(tline)
    %tline = fgetl(fid);
    tlines{end+1,1} = tline;
    tline = fgetl(fid);
        
end
fclose(fid);

for i=1:N_points
tlines_char=char(tlines(i));
coordinates=strsplit(tlines_char,',');
X=cell2mat(coordinates(1));
Y=cell2mat(coordinates(2));
end



%Example: skip the lines




