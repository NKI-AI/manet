function [cropped_new] = edge_detection_and_segmentation(cropped,cropped_old,patch_size)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
format long g;
format compact;
fontSize = 10;

subplot (3,4,1);
hist=histogram(cropped,5);
title('Histogram', 'FontSize', fontSize);

subplot (3,4,2);
imshow(cropped_old,[]);
title('BB cropping', 'FontSize', fontSize);



subplot (3,4,3);
imshow(cropped,[]);
title('Final cropping', 'FontSize', fontSize);

hy = fspecial('sobel');
hx = hy';
Iy = imfilter(double(cropped), hy, 'replicate');
Ix = imfilter(double(cropped), hx, 'replicate');
gradmag = sqrt(Ix.^2 + Iy.^2);
%L = watershed(gradmag);


subplot (3,4,4);
imshow(gradmag,[]);
title('gradmag', 'FontSize', fontSize);
%# Normalize.
g = gradmag - min(gradmag(:));
g = g / max(g(:));
subplot (3,4,5);
imshow(g,[]);
title('Normalized gradmag', 'FontSize', fontSize);

th = graythresh(g); %# Otsu's method.
a = imhmax(g,th/2); %# Conservatively remove local maxima.
th = graythresh(a);
b = a > th; %# Conservative global threshold.

%figure; imshow(b,[]);
title('b', 'FontSize', fontSize);
SE = strel('square', 1)
c = imclose(b,SE); %# Try to close contours.
subplot (3,4,6);
imshow(c,[]);
title('Closing', 'FontSize', fontSize);
d = imfill(c,'holes'); %# Not a bad segmentation by itself.

subplot (3,4,7);
imshow(d,[]);
hold on;
plot(patch_size, patch_size, 'r*', 'LineWidth', 2,'MarkerSize',2);

hold off;
title('Filling:internal markers', 'FontSize', fontSize);
central_ppixel_value=d(patch_size,patch_size); 

if central_ppixel_value==0
 SE2 = strel('square',2)
 d=imclose(d,SE2); %# Try to close contours
 d=imfill(d,'holes');
 subplot (3,4,8);
 imshow(d,[]);
title('Double closing', 'FontSize', fontSize);
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
     
% if d(patch_size,patch_size)==0 | d(patch_size-1,patch_size)==0 | d(patch_size,patch_size-1)==0 | d(patch_size+1,patch_size-1)==0 | d(patch_size+1,patch_size+1)==0
%     dist_internal_marker=internal_marker;
%     subplot (3,4,9); imshow(dist_internal_marker)
%     title('dist internal marker not claculated', 'FontSize', fontSize);
%     
% else    

% title('external markers 2', 'FontSize', fontSize)
% Transform internal marker using distance transform and thresholding.    
dist_internal_marker = bwdist(~internal_marker);
dist_internal_marker(dist_internal_marker <= 1) = 0;
subplot (3,4,9); imshow(dist_internal_marker)
title('dist internal marker', 'FontSize', fontSize);
% Also calculate the distance transform of external marker.
dist_external_marker = bwdist(~external_marker);
dist_external_marker(dist_external_marker <= 1) = 0;
%subplot (3,3,9); imshow(dist_external_marker)


%BW2=ones(1);


%end

cropped_new= zeros(size(cropped), 'logical');
CC = bwconncomp(dist_internal_marker);
numPixels = cellfun(@numel,CC.PixelIdxList);
  
%%simply eliminate the maximum size CC
[biggest,idx] = max(numPixels);
VERIFI=CC.PixelIdxList{idx};
%cropped_new(CC.PixelIdxList{idx}) = 1;

S = regionprops('table',CC,'BoundingBox','Extent','PixelList','Area');
BB=S.BoundingBox;
Pixel_list=S.PixelList;
bboxB = [patch_size-2,patch_size-2,5,5];
Area=S.Area;
[Max_Area,Index_Area]=max(Area);
[Ordered_area,index_order] = sort(Area,'descend') 
Max_index=index_order(1);

values=CC.PixelIdxList{index_order(1)};
overlapRatio = bboxOverlapRatio(BB,bboxB);
[Max_Overlap,Index_overlap]=max(overlapRatio);
if Max_index==Index_overlap
    cropped_new(values) = 1;
else
   
    values=CC.PixelIdxList{Index_overlap};
    %values=CC.PixelIdxList{index_order(2)};
    cropped_new(values) = 1;
    
%     i=2;
%     while i<=size(index_order)
%         
%         if(overlapRatio(i)>0)
%             values=CC.PixelIdxList{index_order(i)};
%             cropped_new(values) = 1;
%             break;
%         else
%             i=i+1;  
%         end
%     end    
    
end
  
%%%%%%%

subplot (3,4,10); imshow(cropped_new)
title('Small objects removed ', 'FontSize', fontSize);
% % 
figure;




end



