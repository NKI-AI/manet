function [dogFilterImage] = DoG3(grayImage)
 %grayImage=im2double(grayImage);
 format long g;
 format compact;
 fontSize = 15;
%DOG Summary of this function goes here
%   Detailed explanation goes here
%grayImage = imread(fullFileName);
% Get the dimensions of the image.  
% numberOfColorBands should be = 1.
[rows columns numberOfColorBands] = size(grayImage);
% Display the original gray scale image.
subplot(5, 2, 1);
imshow(grayImage, []);
title('Input Image', 'FontSize', fontSize);
% Enlarge figure to full screen.
set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
% Give a name to the title bar.
set(gcf,'name','Step by step','numbertitle','off') 
% Let's compute and display the histogram.
[pixelCount grayLevels] = imhist(grayImage);
subplot(5, 2, 2); 
bar(pixelCount);
grid on;
title('Histogram of the Input Image', 'FontSize', fontSize);
xlim([0 grayLevels(end)]); % Scale x axis manually.
%gaussian1 = fspecial('Gaussian', 21, 15);
%gaussian2 = fspecial('Gaussian', 21, 20);
gaussian1 = fspecial('Gaussian', 11, 2);
gaussian2 = fspecial('Gaussian', 5, 1.7);
%dog = gaussian1 - gaussian2;
dogFilterImage1 = conv2(im2double(grayImage), double(gaussian1), 'same');
dogFilterImage2 = conv2(im2double(grayImage), double(gaussian2), 'same');
dogFilterImage=dogFilterImage1-dogFilterImage2;
%dogFilterImage=uint16(dogFilterImage);
subplot(5, 2, 3);
imshow(dogFilterImage, []);
title('DOG Filtered Image', 'FontSize', fontSize);
% Let's compute and display the histogram.
[pixelCount_dog grayLevels_dog] = hist(dogFilterImage(:));
subplot(5, 2, 4); 
bar(grayLevels_dog, pixelCount_dog);
grid on;
title('Histogram of DOG Filtered Image', 'FontSize', fontSize);
%double_grayImage=double(grayImage);


J_median = medfilt2(dogFilterImage);
subplot(5, 2, 5);
imshow(J_median, []);
title('Median Filtered Image', 'FontSize', fontSize);

% J_gamma= imadjust(J_median,[],[],1.25);
% subplot(4, 2, 6);
% imshow(J_gamma, []);
% title('Gamma corrected Image', 'FontSize', fontSize);
J_gamma=J_median;
B = std2(J_gamma);
T=2.5*B;
binary=imbinarize(J_gamma,T);
subplot(5, 2, 7); 
imshow(binary, []);
title('Binarization with fixed T=2.5std', 'FontSize', fontSize);

% loop over all rows and columns
std = std2(J_gamma);
zeroImage = zeros(size(J_gamma,1)-10, size(J_gamma,2)-10);
for ii=6:size(J_gamma,1)-5
    for jj=6:size(J_gamma,2)-5
        % get pixel value5
        pixel=J_gamma(ii,jj);
        %I2 = imcrop(png_image,[X_left Y_left 24 24]);
        %figure; imshow(I2)
        cropped = J_gamma(ii - 2 : ii + 2, jj - 2 : jj + 2, :);
        std_cropped = std2(cropped);
          % check pixel value and assign new value
          if std_cropped < std 
              new_pixel=0;
         else
              new_pixel = 1;
          end
          % save new pixel value in thresholded image
          %image_thresholded(ii,jj)=new_pixel;
          zeroImage(ii,jj)=new_pixel;
      end
end

subplot(5, 2, 8); 
imshow(zeroImage, []);
title('Binarization with varying treshold', 'FontSize', fontSize);
% se = strel('disk',4);
% closeBW = imclose(zeroImage,se);
% figure; imshow(closeBW, []);
% 
% BW2 = imfill(closeBW,'holes');
% figure; imshow(BW2, []);

BW_A = imbinarize(J_gamma, 'adaptive');
subplot(5, 2, 9); 
imshow(BW_A, []);
title('Binarization with adaptive tresholding', 'FontSize', fontSize);


T = graythresh(J_gamma);
BW_T = imbinarize(J_gamma,T);
subplot(5, 2, 10); 
imshow(BW_T, []);
title('Binarization with Otsu', 'FontSize', fontSize);
figure;
end
