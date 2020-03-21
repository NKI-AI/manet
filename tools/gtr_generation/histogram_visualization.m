function [cropped_new] = histogram_visualization(cropped,cropped_old)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%********* FUNZIONANTE***********
% format long g;
% format compact;
% fontSize = 10;
% subplot (3,3,1);
% imshow(cropped,[]);
% title('original', 'FontSize', fontSize);
% 
% hy = fspecial('sobel');
% hx = hy';
% Iy = imfilter(double(cropped), hy, 'replicate');
% Ix = imfilter(double(cropped), hx, 'replicate');
% gradmag = sqrt(Ix.^2 + Iy.^2);
% %L = watershed(gradmag);
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
% %title('b', 'FontSize', fontSize);
% 
% c = imclose(b,ones(1)); %# Try to close contours.
% subplot (3,3,4);
% imshow(c,[]);
% title('Closing', 'FontSize', fontSize);
% d = imfill(c,'holes'); %# Not a bad segmentation by itself.
% subplot (3,3,5);
% imshow(d,[]);
% title('Filling:internal markers', 'FontSize', fontSize);
% 
% % D = bwdist(d);
% % DL = watershed(D);
% % bgm = DL == 0;
% % subplot (3,2,6); imshow(bgm)
% % title('Watershed Ridge Lines')
% %gmag2 = imimposemin(gradmag, bgm | d);
% internal_marker=d;
% %Finally we are ready to compute the watershed-based segmentation.
% se3 = strel('disk', 1);
% external_marker = imopen(internal_marker,se3);
%     
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
% 
% ext_mark2=bwmorph(d,'remove');
% ext_mark2 = bwmorph(ext_mark2,'skel',Inf);
% subplot (3,3,7); imshow(ext_mark2)
% title('external markers 2', 'FontSize', fontSize)
% % Transform internal marker using distance transform and thresholding.    
% dist_internal_marker = bwdist(~internal_marker);
% dist_internal_marker(dist_internal_marker <= 1) = 0;
% subplot (3,3,8); imshow(dist_internal_marker)
% title('dist internal marker', 'FontSize', fontSize);
% % Also calculate the distance transform of external marker.
% dist_external_marker = bwdist(~external_marker);
% dist_external_marker(dist_external_marker <= 1) = 0;
% %subplot (3,3,9); imshow(dist_external_marker)
% %title('dist external marker', 'FontSize', fontSize);
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
% %s = regionprops(dist_internal_marker,'area');
% BW2 = bwareaopen(dist_internal_marker,8);
% %imshow(BW2);
% subplot (3,3,9); imshow(BW2)
% title('Small objects removed ', 'FontSize', fontSize);
% 
% figure;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
format long g;
format compact;
fontSize = 10;

subplot (1,4,1);
hist=histogram(cropped,5);
title('Histogram', 'FontSize', fontSize);


subplot (1,4,2);
imshow(cropped,[]);
title('Final cropping', 'FontSize', fontSize);

hy = fspecial('sobel');
hx = hy';
Iy = imfilter(double(cropped), hy, 'replicate');
Ix = imfilter(double(cropped), hx, 'replicate');
gradmag = sqrt(Ix.^2 + Iy.^2);

subplot (1,4,3);
imshow(gradmag,[]);
title('gradmag', 'FontSize', fontSize);
%# Normalize.
g = gradmag - min(gradmag(:));
g = g / max(g(:));
%subplot (3,4,5);
%imshow(g,[]);
title('Normalized gradmag', 'FontSize', fontSize);
th = graythresh(g); %# Otsu's method.
a = imhmax(g,th/2); %# Conservatively remove local maxima.
th = graythresh(a);
b = a > th; %# Conservative global threshold.

%b = imbinarize(a, 'adaptive')


%figure; imshow(b,[]);
title('b', 'FontSize', fontSize);
SE = strel('square', 1)
c = imclose(b,SE); %# Try to close contours.
%subplot (3,4,6);
%imshow(c,[]);
title('Closing', 'FontSize', fontSize);
d = imfill(c,'holes'); %# Not a bad segmentation by itself.
%subplot (3,4,7);
%imshow(d,[]);
title('Filling:internal markers', 'FontSize', fontSize);
% 

if cropped(size(cropped/2),size(cropped/2))==0
d=imclose(d,SE); %# Try to close contours
d=imclose(d,SE); %# Try to close contours
d=imfill(d,'holes');
end




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

%subplot (3,4,8); imshow(external_marker)
title('external markers', 'FontSize', fontSize)
% 
% ext_mark2=bwmorph(d,'remove');
% ext_mark2 = bwmorph(ext_mark2,'skel',Inf);
% subplot (3,3,7); imshow(ext_mark2)
% title('external markers 2', 'FontSize', fontSize)
% Transform internal marker using distance transform and thresholding.    
dist_internal_marker = bwdist(~internal_marker);
dist_internal_marker(dist_internal_marker <= 1) = 0;
%subplot (3,4,9); imshow(dist_internal_marker)
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
% stats = regionprops('table',dist_internal_marker,'Area');
% CC = bwconncomp(dist_internal_marker);
% areas=stats.Area;
% sizes=size(areas,1);
% aiuto=1;
% if size(areas,1)>1
% max_area=max(areas);
% max_area=max_area-2;
% BW2 = bwareaopen(dist_internal_marker,max_area);
% else
% BW2=dist_internal_marker;
% end
% %imshow(BW2);

cropped_new= zeros(size(cropped), 'logical');
CC = bwconncomp(dist_internal_marker);
numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,idx] = max(numPixels);
cropped_new(CC.PixelIdxList{idx}) = 1;


subplot (1,4,4); imshow(cropped_new)
title('Small objects removed ', 'FontSize', fontSize);
% % 
figure;
%BW2=ones(1);



end