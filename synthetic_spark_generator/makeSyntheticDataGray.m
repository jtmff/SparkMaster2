% SparkMaster 2 (SM2)
% Copyright (C) 2023 Jakub Tomek, jakub.tomek.mff@gmail.com
%
%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <https://www.gnu.org/licenses/>.

% A script that generates synthetic spark data, placing them in /imagesOut. 
% Two subfolders are created there - imgs (actual image files) and masks (binary masks, using red to
% mark where sparks are truly present).


resRows = 1536;
resCols = 1024;
nImages = 2^5;  % How many images are to be generated
noiseSDmin = 25;
noiseSDmax = 45;

% # sparks we want to have
minSparks = 30;
maxSparks = 70;

% # spike couples we want to have
minCouples = 0;
maxCouples = 10;
minCoupleDistCentroids = 25;
maxCoupleDistCentroids = 36;

minDistFromEdge = 55; % minimum distance of centroid from an edge
spikeSqrtFrom = 0.075; % from-to power of spark intensity when embedded. The lower the minimum, the dimmer sparks we get.
spikeSqrtTo = 1.2;

% Parameters for bright and dark background columns.
minBandDistance = 30;
maxBandDistance = 200;
minBandWidth = 5;
maxBandWidth = 25;
bandIntensitySD = 0.075; % in 0-1 multiplier scale - we take a multiplier
bandMinMultiplier = 0.7; % we prevent bands from being too dark or bright
bandMaxMultiplier = 1/0.7;

nSpikeColumnRoulette = [0 0 0 1 1 2];  % we pick a random position here and there will be the corresponding number of columns rich in sparks.
minDistSpikeCol = 50; %min/max dist between two spikes in a column
maxDistSpikeCol = 200;
spikeColDistMin = 200; % minimum distance between spike columns

minDistSparkCentroidsRandom = 80;

% Size scaling
minScaleMultiplierR = 0.8;
maxScaleMultiplierR = 1.2;
minScaleMultiplierC = 0.75;
maxScaleMultiplierC = 1.1;
masScaleMultiplierPairs = 1;

% Description of creation:
% 0) build a library of sparks (these are already stored in sparkLibrary.mat)
% 1) prepare background image:
%    1a) pick a random background intensity between 40 and 80. Add a number of bands (normal distribution with sd of 5? min/max of -20?)
%    1b) add a random profile to all columns
% 2) there will be 0-2 columns rich in sparks (all in-plane or all out-of-plane)
% 3) there will be a number of spark pairs (all in-plane)
% 4) then there will be a number of randomly placed sparks (both in-plane and out-of-plane); they are generated so
% that they're no too close.
% 5) we add noise

% in sparkCentroids{iIm} we store centroids of sparks in an image

%% 1: background image
disp('making bkg');
tic
for iIm = 1:nImages
    % making base
    img = ones(resRows, resCols) * (40 + rand()*40);
    
    % adding bands/stripes
    colStripe = 1 + round(rand()*maxBandDistance);
    while colStripe < resCols
        % add stripe
        stripeMin = colStripe;
        stripeMax = stripeMin + round(5+rand()*(maxBandWidth-minBandWidth));
        if (stripeMax < resCols)
            columnMultiplier = 1 + randn()*bandIntensitySD;
            if (columnMultiplier < bandMinMultiplier)
                columnMultiplier = bandMinMultiplier;
            elseif (columnMultiplier > bandMaxMultiplier)
                columnMultiplier = bandMaxMultiplier;
            end
            img(:, stripeMin:stripeMax) = img(:, stripeMin:stripeMax) * columnMultiplier;
        end
        
        % generate coordinate of the next one
        colStripe = colStripe + round(minBandDistance + rand()*(maxBandDistance-minBandDistance));
    end
    
    % adding profile noise (one row that multiplies every row) - we use half of band intensity sd
    % and we use same min/max as bands/stripes.
    profileNoise = 1 + randn(1, resCols) * (bandIntensitySD/2);
    profileMat = repmat(profileNoise, [resRows, 1]);
    img = img .* profileMat;
    
    
    images{iIm} = img;
    imageMasks{iIm} = zeros(size(img));
end
toc
%% adding spark bands
disp('adding spark bands');
tic
sparklib = load('sparkLibrary');
for iIm = 1:nImages
    img = images{iIm};
    mask = imageMasks{iIm};
    
    colCentroids = [];
    colColumn = [-1000]; % dummy value
    nSpikeCols = nSpikeColumnRoulette(randi(length(nSpikeColumnRoulette)));
    for Col = 1:nSpikeCols
        whichCol = minDistFromEdge + randi(resCols-100);
        distFromOthers = min(abs(whichCol-colColumn));
        while distFromOthers < spikeColDistMin
            whichCol = minDistFromEdge + randi(resCols-100);
            distFromOthers = min(abs(whichCol-colColumn));
        end
        
        colColumn = [colColumn, whichCol];
        
        % populating the column with sparks (the same one)
        if rand()<0.5
            sparks = sparklib.sparksIn;
            sparkMasks = sparklib.masksIn;
        else
            sparks = sparklib.sparksMid;
            sparkMasks = sparklib.masksMid;
        end
        
        
        powerEmbedding = spikeSqrtFrom + rand()*(spikeSqrtTo-spikeSqrtFrom);
        iSparkInLib = randi(length(sparks));
        sparkIntensities = sparks{iSparkInLib} .^ powerEmbedding;
        sparkMask = sparkMasks{iSparkInLib};
        
        resizeScaleR = minScaleMultiplierR+rand()*(maxScaleMultiplierR-minScaleMultiplierR);
        resizeScaleC = minScaleMultiplierC+rand()*(maxScaleMultiplierC-minScaleMultiplierC);
        sparkIntensities = imresize(sparkIntensities, [resizeScaleR*size(sparkMask, 1), resizeScaleC*size(sparkMask, 2)]);
        sparkMask = imresize(sparkMask, [resizeScaleR*size(sparkMask, 1), resizeScaleC*size(sparkMask, 2)]);
        
        rowNext = minDistFromEdge + randi(maxDistSpikeCol);
        while rowNext < (resRows - minDistFromEdge)
            colCentroids = [colCentroids; rowNext, whichCol];
            rowTopLeft = rowNext - round(size(sparkIntensities, 1)/2);
            colTopLeft = whichCol - round(size(sparkIntensities, 1)/2);
            
            % now we multiply the image with the mask
            for iR = rowTopLeft:(rowTopLeft + size(sparkIntensities, 1)-1)
                for iC = colTopLeft:(colTopLeft + size(sparkIntensities, 2)-1)
                    img(iR, iC) = img(iR,iC) * sparkIntensities(iR-rowTopLeft+1, iC-colTopLeft+1);
%                     mask(iR, iC) = sparkMask(iR-rowTopLeft+1, iC-colTopLeft+1);
                    mask(iR, iC) = mask(iR, iC)+sparkMask(iR-rowTopLeft+1, iC-colTopLeft+1);
                end
            end
            rowNext = rowNext + minDistSpikeCol + randi(maxDistSpikeCol - minDistSpikeCol);
        end
    end
    
    % store col centroids in a cell
    imgSparkCentroids{iIm} = colCentroids;
%     figure(1); clf; imshow(uint8(img))
    images{iIm} = img;
    imageMasks{iIm} = mask;
end
toc
%% 3 adding spark couples
disp('adding spark couples');
tic
allSparks = [sparklib.sparksIn, sparklib.sparksMid];
allSparks = allSparks(1:5); % we remove the last massive spark which would give strange visuals.
allSparkMasks = [sparklib.masksIn, sparklib.masksMid];
allSparkMasks = allSparkMasks(1:5);
for iIm = 1:nImages
    centroids = imgSparkCentroids{iIm};
    img = images{iIm};
    mask = imageMasks{iIm};
    
    toRemoveFirstRowLater = false;
    if isempty(centroids) % this is purely technical so that the code does not crash if there are no prior centroids.
        centroids = [-1e6, -1e6];
        toRemoveFirstRowLater = true;
    end
    
    
    toMake = randi(maxCouples+1) - 1;
    
    for iSpark = 1:toMake
        % picking the location of the first spark
        sparkRow = 2*minDistFromEdge + randi(resRows - 4*minDistFromEdge);
        sparkCol = 2*minDistFromEdge + randi(resCols - 4*minDistFromEdge);
        minDistToOthers = 2* min(pdist2([sparkRow sparkCol], centroids)); %2x because we require a bit larger distance.

        maxAttempts = 1000; % if we fail this number of times, we give up
        % if too close, we try another location.
        while minDistToOthers < minDistSparkCentroidsRandom && maxAttempts > 0
            sparkRow = 2*minDistFromEdge + randi(resRows - 4*minDistFromEdge);
            sparkCol = 2*minDistFromEdge + randi(resCols - 4*minDistFromEdge);
            minDistToOthers = min(pdist2([sparkRow sparkCol], centroids));
            maxAttempts = maxAttempts - 1;
        end
        if maxAttempts == 0 % no point adding other spikes, the field is too populated
            break;
        end
        
        rho = minCoupleDistCentroids + randi(maxCoupleDistCentroids - minCoupleDistCentroids); % distance
        theta = rand()*2*pi;
        [offsetSecondRows, offsetSecondCols] = pol2cart(theta,rho);
        sparkSecondRow = round(sparkRow + offsetSecondRows);
        sparkSecondCol = round(sparkCol + offsetSecondCols);
        centroids = [centroids; sparkRow, sparkCol; sparkSecondRow, sparkSecondCol];
        
        % adding first spark
        powerEmbedding = spikeSqrtFrom + rand()*(spikeSqrtTo-spikeSqrtFrom);
        
        iSparkInLib = randi(length(allSparks));
        sparkIntensities = allSparks{iSparkInLib} .^ powerEmbedding;
        sparkMask = allSparkMasks{iSparkInLib};
        
        resizeScaleR = minScaleMultiplierR+rand()*(maxScaleMultiplierR-minScaleMultiplierR);
        resizeScaleC = minScaleMultiplierC+rand()*(maxScaleMultiplierC-minScaleMultiplierC);
        sparkIntensities = imresize(sparkIntensities, [resizeScaleR*size(sparkMask, 1), resizeScaleC*size(sparkMask, 2)]);
        sparkMask = imresize(sparkMask, [resizeScaleR*size(sparkMask, 1), resizeScaleC*size(sparkMask, 2)]);
        
        rowTopLeft = sparkRow - round(size(sparkIntensities, 1)/2);
        colTopLeft = sparkCol - round(size(sparkIntensities, 1)/2);
        for iR = rowTopLeft:(rowTopLeft + size(sparkIntensities, 1)-1)
            for iC = colTopLeft:(colTopLeft + size(sparkIntensities, 2)-1)
                img(iR, iC) = img(iR,iC) * sparkIntensities(iR-rowTopLeft+1, iC-colTopLeft+1);
%                 mask(iR, iC) = sparkMask(iR-rowTopLeft+1, iC-colTopLeft+1);
                mask(iR, iC) = mask(iR, iC)+sparkMask(iR-rowTopLeft+1, iC-colTopLeft+1);
            end
        end
        
        % and the second - TODO use the same power?
        % nonelegant, should be for loop along with reading the first spark really...
        powerEmbedding = spikeSqrtFrom + rand()*(spikeSqrtTo-spikeSqrtFrom);
        
        iSparkInLib = randi(length(allSparks));
        sparkIntensities = allSparks{iSparkInLib} .^ powerEmbedding;
        sparkMask = allSparkMasks{iSparkInLib};
        
        resizeScaleR = minScaleMultiplierR+rand()*(maxScaleMultiplierR-minScaleMultiplierR);
        resizeScaleC = minScaleMultiplierC+rand()*(maxScaleMultiplierC-minScaleMultiplierC);
        sparkIntensities = imresize(sparkIntensities, [resizeScaleR*size(sparkMask, 1), resizeScaleC*size(sparkMask, 2)]);
        sparkMask = imresize(sparkMask, [resizeScaleR*size(sparkMask, 1), resizeScaleC*size(sparkMask, 2)]);
        
        rowTopLeft = sparkSecondRow - round(size(sparkIntensities, 1)/2);
        colTopLeft = sparkSecondCol - round(size(sparkIntensities, 1)/2);
        for iR = rowTopLeft:(rowTopLeft + size(sparkIntensities, 1)-1)
            for iC = colTopLeft:(colTopLeft + size(sparkIntensities, 2)-1)
                img(iR, iC) = img(iR,iC) * sparkIntensities(iR-rowTopLeft+1, iC-colTopLeft+1);
%                 mask(iR, iC) = sparkMask(iR-rowTopLeft+1, iC-colTopLeft+1);
                mask(iR, iC) = mask(iR, iC)+sparkMask(iR-rowTopLeft+1, iC-colTopLeft+1);
            end
        end
    end
    
    
    % now we have the centroid of the first spike that we'll add
    if (toRemoveFirstRowLater)
        centroids(1,:) = [];
    end
    imgSparkCentroids{iIm} = centroids;
    images{iIm} = img;
%     figure(2); clf; imshow(uint8(img))
    imageMasks{iIm} = mask;
end
toc
%% 4 adding random sparks
disp('adding random sparks');
tic
% make sure not too close to the edges - at least 40 away?
allSparks = [sparklib.sparksIn, sparklib.sparksMid];
allSparkMasks = [sparklib.masksIn, sparklib.masksMid];
for iIm = 1:nImages
    centroids = imgSparkCentroids{iIm};
    img = images{iIm};
    mask = imageMasks{iIm};
    
    toRemoveFirstRowLater = false;
    if isempty(centroids) % this is purely technical so that the code does not crash if there are no prior centroids.
        centroids = [-1e6, -1e6];
        toRemoveFirstRowLater = true;
    end
    
    targetSparkN = minSparks + randi(maxSparks-minSparks);
    toMake = targetSparkN - size(centroids, 1);
    
    for iSpark = 1:toMake
        sparkRow = minDistFromEdge + randi(resRows - 2*minDistFromEdge);
        sparkCol = minDistFromEdge + randi(resCols - 2*minDistFromEdge);
        minDistToOthers = min(pdist2([sparkRow sparkCol], centroids));
        
        
        maxAttempts = 1000; % if we fail this number of times, we give up
        % if too close, we try another location.
        while minDistToOthers < minDistSparkCentroidsRandom && maxAttempts > 0
            sparkRow = minDistFromEdge + randi(resRows - 2*minDistFromEdge);
            sparkCol = minDistFromEdge + randi(resCols - 2*minDistFromEdge);
            minDistToOthers = min(pdist2([sparkRow sparkCol], centroids));
            maxAttempts = maxAttempts - 1;
        end
        if maxAttempts == 0 % no point adding other spikes, the field is too populated
            break;
        end
        
        % here we have good coordinates, so we randomly select and add a spark.
        centroids = [centroids; sparkRow, sparkCol];
               
        % now we multiply the image with the mask
        powerEmbedding = spikeSqrtFrom + rand()*(spikeSqrtTo-spikeSqrtFrom);
        iSparkInLib = randi(length(allSparks));
        sparkIntensities = allSparks{iSparkInLib} .^ powerEmbedding;
        sparkMask = allSparkMasks{iSparkInLib};
        
        resizeScaleR = minScaleMultiplierR+rand()*(maxScaleMultiplierR-minScaleMultiplierR);
        resizeScaleC = minScaleMultiplierC+rand()*(maxScaleMultiplierC-minScaleMultiplierC);
        sparkIntensities = imresize(sparkIntensities, [resizeScaleR*size(sparkMask, 1), resizeScaleC*size(sparkMask, 2)]);
        sparkMask = imresize(sparkMask, [resizeScaleR*size(sparkMask, 1), resizeScaleC*size(sparkMask, 2)]);
        
        rowTopLeft = sparkRow - round(size(sparkIntensities, 1)/2);
        colTopLeft = sparkCol - round(size(sparkIntensities, 1)/2);
        for iR = rowTopLeft:(rowTopLeft + size(sparkIntensities, 1)-1)
            for iC = colTopLeft:(colTopLeft + size(sparkIntensities, 2)-1)
                img(iR, iC) = img(iR,iC) * sparkIntensities(iR-rowTopLeft+1, iC-colTopLeft+1);
                mask(iR, iC) = mask(iR, iC)+sparkMask(iR-rowTopLeft+1, iC-colTopLeft+1);
            end
        end
    end
       
    
    if (toRemoveFirstRowLater)
        centroids(1,:) = [];
    end
    imgSparkCentroids{iIm} = centroids;
    images{iIm} = img;
%     figure(3); clf; imshow(uint8(img))
    imageMasks{iIm} = mask;
end
toc
%% 5: adding noise and saving
disp('adding noise and saving');
mkdir imagesOut/imgs
mkdir imagesOut/masks
tic


lenNumericFname = 4; %'0001.png' etc.

for iIm = 1:nImages
    
    noiseSD = noiseSDmin + rand()*(noiseSDmax - noiseSDmin); % noise stdev is uniformly distributed between min and max
    
    images{iIm} = uint8(images{iIm} + randn(resRows, resCols) * noiseSD);
    
%     im = 
%     figure(4); 
%     subplot(1,2,1); imshow(images{iIm});
%     subplot(1,2,2); imshow(imageMasks{iIm});
    maskRGB = zeros(size(imageMasks{1}, 1), size(imageMasks{1}, 2), 3);
    maskRGB(:,:,1) = imageMasks{iIm}*255;
    maskRGB = uint8(maskRGB);
    
    nZerosToAdd = lenNumericFname - length(num2str(iIm));
    numString = '';
    while length(numString) < nZerosToAdd
        numString = [numString '0'];
    end
    
    imwrite(images{iIm}, ['imagesOut/imgs/' numString num2str(iIm) '.png']);
    imwrite(maskRGB, ['imagesOut/masks/' numString num2str(iIm) '.png']);
end
toc
