function [max_numberOFpixels,min_numberOFpixels,mean_numberOFpixels,mask_for_visulatization] = roi_generation5(coordinates,image,spacing,mask_for_visulatization,file_name,out_dir,patch_size,bbox_size,pixel_size,count)
%ROI_GENERATION Summary of this function goes here
%   Detailed explanation goes here
 format long g;
 format compact;
 fontSize = 10;
 patch_size_for_visualization=8;
 len=size(coordinates,1);
 num_pixels_vector = cell(0,1);
 %imshow(image,[]);
 for i=1:len
  X_ph=coordinates(i,1);
  Y_ph=coordinates(i,2);
  X_pix=round(X_ph*[pixel_size/spacing]);
  Y_pix=round(Y_ph*[pixel_size/spacing]);

  disp('Y_pix')
  disp(Y_pix)
  disp('X_pix')
  disp(X_pix)
  
  if (Y_pix - patch_size<0 | X_pix - patch_size <0 )
      count=count+1
      if (Y_pix - patch_size<0)
          disp('Y_pix - patch_size')
          disp(Y_pix - patch_size)
          padding=ceil(patch_size-Y_pix)
          patch_size=patch_size-padding-1
          patch_size_for_visualization=patch_size
          %addition=zeros(padding,size(image,2));
          %image=[addition;image];
          %cropped = image((Y_pix + padding+1)- patch_size : (Y_pix+ padding+1)+ patch_size, X_pix - patch_size : X_pix + patch_size, :);
          %disp((Y_pix+ padding) - patch_size)  
      end    
      if (X_pix - patch_size<0)   
          disp('X_pix - patch_size')
          disp(X_pix - patch_size)
          padding=ceil(patch_size-X_pix)
          patch_size=patch_size-padding-1
          patch_size_for_visualization=patch_size
          %addition=zeros(size(image,1),padding);
          %image=[addition,image];
          %disp(X_pix)
          %disp(size(image,2))
          %size(image)
          %cropped = image(Y_pix - patch_size : Y_pix + patch_size, (X_pix+ padding+1) - patch_size : (X_pix+ padding+1) + patch_size , :);
          
      end
      %continue
  end    
  cropped = image(Y_pix - patch_size : Y_pix + patch_size, X_pix - patch_size : X_pix + patch_size, :);
  
  %%%%FIND THE NEW CENTER BY USING THE BBOX
    max_value_bb=image(Y_pix,X_pix);
    value_to_debugY=Y_pix - bbox_size;
    value_to_debugX=X_pix - bbox_size;
    value_to_debugY2=Y_pix + bbox_size;
    value_to_debugX2=X_pix + bbox_size;
    
    
    bbox= image(Y_pix - bbox_size : Y_pix + bbox_size, X_pix - bbox_size : X_pix + bbox_size, :);
    bbox_leftup_X=X_pix - bbox_size;
    bbox_leftup_Y=Y_pix - bbox_size;
    bbox_Rup_X=X_pix + bbox_size;
    bbox_Rup_Y=Y_pix + bbox_size;
    
    center_points_bb=find_new_center_bbox(image,max_value_bb,X_pix,Y_pix,bbox_size);
    %center_points_bb=find_new_center_circle(image,pline_x,pline_y,max_value,X_pix,Y_pix);
    
    center_points_mat_bb=cell2mat(center_points_bb);
    if center_points_mat_bb(1,3)==1
        image2=image;
        
        X_pix_new_bb=center_points_mat_bb(1,1);
        Y_pix_new_bb=center_points_mat_bb(1,2);
        
        image2=image;
        image2(Y_pix_new_bb,X_pix_new_bb)=0;
        
        center_points_bb=find_new_center_bbox(image2,max_value_bb,X_pix,Y_pix,bbox_size);
        
        if center_points_mat_bb(1,3)==1
            
        %X_pix_new_bb=center_points_mat_bb(1,1);
        %Y_pix_new_bb=center_points_mat_bb(1,2);    
            
        disp('new center founded')   
        cropped_new_bb = image(Y_pix_new_bb - patch_size : Y_pix_new_bb + patch_size, X_pix_new_bb - patch_size : X_pix_new_bb + patch_size, :);
        cropped_new_bb_larger = image(Y_pix_new_bb - patch_size_for_visualization : Y_pix_new_bb + patch_size_for_visualization, X_pix_new_bb - patch_size_for_visualization : X_pix_new_bb + patch_size_for_visualization, :);

        end

    else

      cropped_new_bb=cropped;  
      cropped_new_bb_larger=cropped;
      X_pix_new_bb=X_pix;
      Y_pix_new_bb=Y_pix;

         
    end    
  
    [binarization,Area]=edge_detection_vs_gaussian5(cropped_new_bb,cropped_new_bb_larger,patch_size);
    

    image_height=size(mask_for_visulatization,1);
    image_width=size(mask_for_visulatization,2);
    mask_image=zeros(image_height,image_width,1,'single');    
    mask_image(Y_pix_new_bb - patch_size : Y_pix_new_bb + patch_size, X_pix_new_bb - patch_size : X_pix_new_bb + patch_size, :)=binarization;
    mask_for_visulatization=mask_image|mask_for_visulatization;

    
    num_pixels_vector{i,1} = Area;
   
   
 end
out_file=strcat(out_dir,file_name);
%savefig(out_file); %saves the current figure to a FIG-file named filename.fig.  
%figure;

 num_pixels_vector_mat= cell2mat(num_pixels_vector);
 max_numberOFpixels=max(num_pixels_vector_mat);
 min_numberOFpixels=min(num_pixels_vector_mat);
 mean_numberOFpixels=mean(num_pixels_vector_mat);
 
 
 
 
end

