function[] = apply_watershed(Image)
format long g;
format compact;
fontSize = 20;

hy = fspecial('sobel');
hx = hy';
Iy = imfilter(double(Image), hy, 'replicate');
Ix = imfilter(double(Image), hx, 'replicate');
gradmag = sqrt(Ix.^2 + Iy.^2);
L = watershed(gradmag);


% figure; imshow(gradmag,[]);
% title('gradmag', 'FontSize', fontSize);
% %# Normalize.
% g = gradmag - min(gradmag(:));
% g = g / max(g(:));
% figure; imshow(g,[]);
% title('Normalized gradmag', 'FontSize', fontSize);
% th = graythresh(g); %# Otsu's method.
% a = imhmax(g,th/2); %# Conservatively remove local maxima.
% th = graythresh(a);
% b = a > th/4; %# Conservative global threshold.
% figure; imshow(b,[]);
% title('b', 'FontSize', fontSize);
% 
% c = imclose(b,ones(1)); %# Try to close contours.
% figure; imshow(c,[]);
% title('c', 'FontSize', fontSize); %%C è una buona
% d = imfill(c,'holes'); %# Not a bad segmentation by itself.
% figure; imshow(d,[]);
% title('d', 'FontSize', fontSize);



% %# Use the rough segmentation to define markers.
% g2 = imimposemin(g, ~ imdilate( bwperim(a), ones(1)),4 );
% 
% L = watershed(g2);
% figure; imshow(L,[]);
% title('L', 'FontSize', fontSize);

subplot(2, 2, 1);
imshow(Image, []);
title('Original Grayscale Image', 'FontSize', fontSize);

subplot(2, 2, 2);
imshow(gradmag, []);
title('Gradient magnitude  Image', 'FontSize', fontSize);


subplot(2, 2, 3);
imshow(L, []);
title('Watershed Image', 'FontSize', fontSize);

bw_mask = logical(L==2);
subplot(2, 2, 4);
imshow(bw_mask, []);
title('Binary Image', 'FontSize', fontSize);


end

