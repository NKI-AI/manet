function [thisClass] = clustering(spectralImage)
  format long g;
  format compact;
  fontSize = 25;
  %===============================================================================
  subplot(1,2,1);
  imshow(spectralImage);
  caption = sprintf('Original Image', 1);
  title(caption, 'FontSize', fontSize);
  
  drawnow; % Force display to update immediately. 
  % It is supposed to be monochrome because it's a spectral image. Get the dimensions of the image.
  % numberOfColorChannels should be = 1 for a gray scale image, and 3 for an RGB color image.
  [rows, columns, numberOfColorChannels] = size(spectralImage);
  
  if numberOfColorChannels > 1
	  % It's not really gray scale like we expected - it's color.
	  % Use weighted sum of ALL channels to create a gray scale image.
	  spectralImage = rgb2gray(spectralImage);
	  % ALTERNATE METHOD: Convert it to gray scale by taking only the green channel,
	  % which in a typical snapshot will be the least noisy channel.
	  % grayImage = grayImage(:, :, 2); % Take green channel.
    % Display image.
  end 
  format long g;
  format compact;
  fontSize = 10;
  subplot (3,3,1);
  imshow(spectralImage,[]);
  title('original', 'FontSize', fontSize);

  hy = fspecial('sobel');
  hx = hy';
  Iy = imfilter(double(spectralImage), hy, 'replicate');
  Ix = imfilter(double(spectralImage), hx, 'replicate');
  gradmag = sqrt(Ix.^2 + Iy.^2);
  %L = watershed(gradmag);

 subplot (3,3,2);
 imshow(gradmag,[]);
 title('gradmag', 'FontSize', fontSize);
 %# Normalize.
 g = gradmag - min(gradmag(:));
 g = g / max(g(:));
 subplot (3,3,3);
 imshow(g,[]);
 title('Normalized gradmag', 'FontSize', fontSize);
  
 spectralImage=g;
  
  k=1;
  if k == 1
  	 data = double(spectralImage(:));
  else
  	 data = [data, double(spectralImage(:))];
  end
  
  % Enlarge figure to half screen.
  set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.5, 0.96]);
  drawnow;
  hp = impixelinfo();
  % %----------------------------------------------------------------------------------------
  numberOfClasses = 2; % Assume 2.
  %----------------------------------------------------------------------------------------
  % KMEANS CLASSIFICATION RIGHT HERE!!!
  % Each row of data represents one pixel.  Each column of data represents one image.
  % Have kmeans decide which cluster each pixel belongs to.
  indexes = kmeans(data, numberOfClasses);
  %----------------------------------------------------------------------------------------
  % Let's convert what class index the pixel is into images for each class index.
  % Assume 2 clusters.
  class1 = reshape(indexes == 1, rows, columns);
  class2 = reshape(indexes == 2, rows, columns);

  % Let's put these into a 3-D array for later to make it easy to display them all with a loop.
  allClasses = cat(3, class1, class2);
  allClasses = allClasses(:, :, 1:numberOfClasses); % Crop off just what we need.
  % OK!  WE'RE ALL DONE!.  Nothing left now but to display our classification images.
  plotRows = ceil(sqrt(size(allClasses, 3)));
  % Display the classes, both binary and masking the original.
  % Also make an indexes image so we can display each class in a unique color.
  indexedImageK = zeros(rows, columns, 'uint16'); % Initialize another indexed image.
  
  % Display binary image of what pixels have this class ID number.
  subplot(3, 3, 4);
  thisClass = allClasses(:, :, 2);
  imshow(thisClass);
  caption = sprintf('Binarized Image', 1);
  title(caption, 'FontSize', fontSize);
  figure;
  
  % Enlarge figure to full screen.
  %set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.5, 0.04, 0.5, 0.96]);
  %message = sprintf('Done with Classification');
  %helpdlg(message);
  
  
  

end

