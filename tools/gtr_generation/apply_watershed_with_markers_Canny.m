function [BW2] = apply_watershed_with_markers_Canny(cropped)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% format long g;
% format compact;
% fontSize = 10;
% subplot (3,3,1);
% imshow(cropped,[]);
% title('original', 'FontSize', fontSize);
% 
% % hy = fspecial('sobel');
% % hx = hy';
% % Iy = imfilter(double(cropped), hy, 'replicate');
% % Ix = imfilter(double(cropped), hx, 'replicate');
% % gradmag = sqrt(Ix.^2 + Iy.^2);
% %L = watershed(gradmag);
% gradmag = edge(cropped,'Canny');
% 
% subplot (3,3,2);
% imshow(gradmag,[]);
% title('gradmag', 'FontSize', fontSize);
% %# Normalize.
% g = gradmag - min(gradmag(:));
% g = g / max(g(:));
% subplot (3,3,3);
% imshow(g,[]);
% title('Normalized gradmag', 'FontSize', fontSize);
% th = graythresh(g); %# Otsu's method.
% a = imhmax(g,th/2); %# Conservatively remove local maxima.
% th = graythresh(a);
% b = a > th; %# Conservative global threshold.
% %figure; imshow(b,[]);
% title('b', 'FontSize', fontSize);
% SE = strel('square', 4)
% c = imclose(b,SE); %# Try to close contours.
% subplot (3,3,4);
% imshow(c,[]);
% title('Closing', 'FontSize', fontSize);
% d = imfill(c,'holes'); %# Not a bad segmentation by itself.
% subplot (3,3,5);
% imshow(d,[]);
% title('Filling:internal markers', 'FontSize', fontSize);
% % 
% 
% internal_marker=d;
% %Finally we are ready to compute the watershed-based segmentation.
% se3 = strel('disk', 1);
% external_marker = imopen(internal_marker,se3);
% %     
% se4 = strel('disk', 1);
% external_marker = imdilate(external_marker,se4);
%     
% % Then using internal marker, obtain the external marker.
% external_marker = ~internal_marker & external_marker;
% 
% %external_marker = bwmorph(external_marker,'skel',Inf);
% 
% subplot (3,3,6); imshow(external_marker)
% title('external markers', 'FontSize', fontSize)
% % 
% % ext_mark2=bwmorph(d,'remove');
% % ext_mark2 = bwmorph(ext_mark2,'skel',Inf);
% % subplot (3,3,7); imshow(ext_mark2)
% % title('external markers 2', 'FontSize', fontSize)
% % Transform internal marker using distance transform and thresholding.    
% dist_internal_marker = bwdist(~internal_marker);
% dist_internal_marker(dist_internal_marker <= 1) = 0;
% subplot (3,3,7); imshow(dist_internal_marker)
% title('dist internal marker', 'FontSize', fontSize);
% % Also calculate the distance transform of external marker.
% dist_external_marker = bwdist(~external_marker);
% dist_external_marker(dist_external_marker <= 1) = 0;
% %subplot (3,3,9); imshow(dist_external_marker)
% %title('dist external marker', 'FontSize', fontSize);
% % 
% % 
% % % Combine markers for Watershed algorithm.    
% % markers = internal_marker + external_marker;
% % final_img = imimposemin(gradmag, dist_external_marker | dist_internal_marker);
% % %gray_img = imimposemin(gradmag, markers);
% % WSL = watershed(final_img);
% % % subplot (3,3,9); %imshow(WSL)
% % % title('Watershed with external 1', 'FontSize', fontSize);
% % % imshow(WSL,[]);
% % % figure;
% % 
% % bw_mask = logical(WSL==2);
% % %imshow(bw_mask,[]);
% % %figure;
% % final_img2 = imimposemin(gradmag, ext_mark2 | internal_marker);
% % %gray_img = imimposemin(gradmag, markers);
% % WSL2 = watershed(final_img2);
% % %subplot (3,3,9); imshow(WSL2)
% % %title('Watershed with external 2', 'FontSize', fontSize);
% % 
% % %CC = bwconncomp(mask_image)
% %s = regionprops(dist_internal_marker,'area');
% %BW2 = bwareaopen(dist_internal_marker,8);
% BW2 = bwareaopen(dist_internal_marker,3);
% %imshow(BW2);
% subplot (3,3,8); imshow(BW2)
% title('Small objects removed ', 'FontSize', fontSize);
% % % 
% figure;
% %BW2=ones(1);

format long g;
format compact;
fontSize = 10;
subplot (3,3,1);
imshow(cropped,[]);
title('original', 'FontSize', fontSize);
B = imgaussfilt(cropped);
subplot (3,3,2);
imshow(B,[]);
title('Gaussian filter', 'FontSize', fontSize);
gradmag = edge(cropped,'Canny');
subplot (3,3,3);
imshow(gradmag,[]);
title('gradmag', 'FontSize', fontSize);


%figure; imshow(b,[]);
title('b', 'FontSize', fontSize);
SE = strel('square', 4)
c = imclose(gradmag,SE); %# Try to close contours.
subplot (3,3,3);
imshow(c,[]);
title('Closing', 'FontSize', fontSize);
d = imfill(c,'holes'); %# Not a bad segmentation by itself.
subplot (3,3,4);
imshow(d,[]);
title('Filling:internal markers', 'FontSize', fontSize);
% 

internal_marker=d;
%Finally we are ready to compute the watershed-based segmentation.
se3 = strel('disk', 1);
external_marker = imopen(internal_marker,se3);
%     
se4 = strel('disk', 1);
external_marker = imdilate(external_marker,se4);
    
% Then using internal marker, obtain the external marker.
external_marker = ~internal_marker & external_marker;

%external_marker = bwmorph(external_marker,'skel',Inf);

subplot (3,3,5); imshow(external_marker)
title('external markers', 'FontSize', fontSize)
% 
% ext_mark2=bwmorph(d,'remove');
% ext_mark2 = bwmorph(ext_mark2,'skel',Inf);
% subplot (3,3,7); imshow(ext_mark2)
% title('external markers 2', 'FontSize', fontSize)
% Transform internal marker using distance transform and thresholding.    
dist_internal_marker = bwdist(~internal_marker);
dist_internal_marker(dist_internal_marker <= 1) = 0;
subplot (3,3,6); imshow(dist_internal_marker)
title('dist internal marker', 'FontSize', fontSize);
% Also calculate the distance transform of external marker.
dist_external_marker = bwdist(~external_marker);
dist_external_marker(dist_external_marker <= 1) = 0;
%subplot (3,3,9); imshow(dist_external_marker)
%title('dist external marker', 'FontSize', fontSize);
% 
% 
% % Combine markers for Watershed algorithm.    
% markers = internal_marker + external_marker;
% final_img = imimposemin(gradmag, dist_external_marker | dist_internal_marker);
% %gray_img = imimposemin(gradmag, markers);
% WSL = watershed(final_img);
% % subplot (3,3,9); %imshow(WSL)
% % title('Watershed with external 1', 'FontSize', fontSize);
% % imshow(WSL,[]);
% % figure;
% 
% bw_mask = logical(WSL==2);
% %imshow(bw_mask,[]);
% %figure;
% final_img2 = imimposemin(gradmag, ext_mark2 | internal_marker);
% %gray_img = imimposemin(gradmag, markers);
% WSL2 = watershed(final_img2);
% %subplot (3,3,9); imshow(WSL2)
% %title('Watershed with external 2', 'FontSize', fontSize);
% 
% %CC = bwconncomp(mask_image)
%s = regionprops(dist_internal_marker,'area');
%BW2 = bwareaopen(dist_internal_marker,8);
%BW2 = bwareaopen(dist_internal_marker,3);
s = regionprops(BW2,'Area');
%imshow(BW2);
subplot (3,3,7); imshow(BW2)
title('Small objects removed ', 'FontSize', fontSize);
% % 
figure;
%BW2=ones(1);







end
