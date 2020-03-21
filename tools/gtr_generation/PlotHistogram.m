%=====================================================================
% Plots the histogram of imgArray in axes axesImage
% If imgArray is a double, it must be normalized between 0 and 1.
function [minGL, maxGL ,gl1Percentile ,gl99Percentile] =PlotHistogram(imgArray)
% Get a histogram of the entire image.
% Use 1024 bins.
[COUNTS, GLs] = imhist(imgArray, 256); % make sure you label theaxes after imhist because imhist will destroy them.
% GLs goes from 0 (at element 1) to 256 (at element 1024) but only afraction of
% these bins have data in them. The upper ones are generally 0.Find the last
% non-zero bin so we can plot just up to there to get betterhorizontal resolution.
maxBinUsed = max(find(COUNTS));
% Get subportion of array that has non-zero data.
COUNTS = COUNTS(1:maxBinUsed);
GLs = GLs(1:maxBinUsed);
% The first bin is not meaningful image data and just wrecks thescale of the
% histogram plot, so zero that one out. This is because it's a huge
% spike at zero due to masking.
COUNTS(1) = 0;
 
minBinUsed = min(find(COUNTS));
if isempty(minBinUsed)
    % No pixels were selected - the entire image is zero.
    minGL = GLs(maxBinUsed);
    maxGL = GLs(maxBinUsed);
    minBinUsed = maxBinUsed;
else
    % There is some spread to the histogram.
    minGL = GLs(minBinUsed);
    maxGL = GLs(maxBinUsed);
end
 
summed = sum(COUNTS);
cdf = 0;
gl1Percentile = minGL;   % Need in case the first bin is more than 1%in which case the if below will never get entered.
for bin = minBinUsed : maxBinUsed
    cdf = cdf + COUNTS(bin);
    if cdf < 0.01 * summed         gl1Percentile = GLs(bin);     end     if cdf > 0.99 * summed
        break;
    end
end
gl99Percentile = GLs(bin);
 
return; % PlotHistogram
end