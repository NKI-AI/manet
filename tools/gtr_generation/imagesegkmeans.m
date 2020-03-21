function [] =imagesegkmeans(image)
%K_ Summary of this function goes here
%   Detailed explanation goes here
wavelength = 2.^(0:5) * 3;
orientation = 0:45:135;
g = gabor(wavelength,orientation);

gabormag = imgaborfilt(image,g);
montage(gabormag,'Size',[4 6])

for i = 1:length(g)
    sigma = 0.5*g(i).Wavelength;
    gabormag(:,:,i) = imgaussfilt(gabormag(:,:,i),3*sigma,'FilterSize',1); 
end
montage(gabormag,'Size',[4 6])

nrows = size(image,1);
ncols = size(image,2);
[X,Y] = meshgrid(1:ncols,1:nrows);
featureSet = cat(3,image,gabormag,X,Y);
L2 = imsegkmeans(featureSet,2,'NormalizeInput',true,'NumAttempts',5);
%L2 = imsegkmeans(image,2,'NormalizeInput',true,'NumAttempts',5)
C = labeloverlay(image,L2,'Colormap',[0, 0, 0; 1, 1, 1]);


numberOfClasses=2;
[rows, columns, numberOfColorChannels] = size(image);
class1 = reshape(L2 == 1, rows, columns);
class2 = reshape(L2 == 2, rows, columns);

% Let's put these into a 3-D array for later to make it easy to display them all with a loop.
allClasses = cat(3, class1, class2);
allClasses = allClasses(:, :, 1:numberOfClasses); % Crop off just what we need.
% OK!  WE'RE ALL DONE!.  Nothing left now but to display our classification images.
plotRows = ceil(sqrt(size(allClasses, 3)));
% Display the classes, both binary and masking the original.
% Also make an indexes image so we can display each class in a unique color.
indexedImageK = zeros(rows, columns, 'uint16'); % Initialize another indexed image.
  
% Display binary image of what pixels have this class ID number.


subplot(1,3,1)
imshow(image,[])
subplot(1,3,2)
imshow(C,[])
title('Labeled Image')
subplot(1,3,3);
thisClass = allClasses(:, :, 2);
imshow(thisClass);
caption = sprintf('Binarized Image', 1);






figure;
end

