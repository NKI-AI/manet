function [dogFilterImage] = DoG(grayImage)
 format long g;
 format compact;
 fontSize = 25;
%DOG Summary of this function goes here
%   Detailed explanation goes here
%grayImage = imread(fullFileName);
% Get the dimensions of the image.  
% numberOfColorBands should be = 1.
[rows columns numberOfColorBands] = size(grayImage);
% Display the original gray scale image.
subplot(3, 2, 1);
imshow(grayImage, []);
title('Original Grayscale Image', 'FontSize', fontSize);
% Enlarge figure to full screen.
set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
% Give a name to the title bar.
set(gcf,'name','Demo by ImageAnalyst','numbertitle','off') 
% Let's compute and display the histogram.
[pixelCount grayLevels] = imhist(grayImage);
subplot(3, 2, 2); 
bar(pixelCount);
grid on;
title('Histogram of Original Image', 'FontSize', fontSize);
xlim([0 grayLevels(end)]); % Scale x axis manually.
%gaussian1 = fspecial('Gaussian', 21, 15);
%gaussian2 = fspecial('Gaussian', 21, 20);
gaussian1 = fspecial('Gaussian', 21, 6.5);
gaussian2 = fspecial('Gaussian', 21, 5);
dog = gaussian1 - gaussian2;
dogFilterImage = conv2(im2double(grayImage), double(dog), 'same');
%dogFilterImage=uint16(dogFilterImage);
subplot(3, 2, 3);
imshow(dogFilterImage, []);
title('DOG Filtered Image', 'FontSize', fontSize);
% Let's compute and display the histogram.
[pixelCount_dog grayLevels_dog] = hist(dogFilterImage(:));
subplot(3, 2, 4); 
bar(grayLevels_dog, pixelCount_dog);
grid on;
title('Histogram of DOG Filtered Image', 'FontSize', fontSize);
%double_grayImage=double(grayImage);
final_image=imsubtract(im2double(grayImage),dogFilterImage);


subplot(3, 2, 5); 
imshow(final_image, []);
title('Subtraction Image', 'FontSize', fontSize);
[pixelCount_sub grayLevels_sub] = hist(final_image(:));
subplot(3, 2, 6); 
bar(grayLevels_sub, pixelCount_sub);
grid on;
title('Histogram of DOG Filtered Image', 'FontSize', fontSize);

% final_image_uint=uint16(final_image);
% subplot (2,1,1)
% imshow(final_image_uint, []);
% subplot (2,1,2)
% imshow(final_image, []);


end

