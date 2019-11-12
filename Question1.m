%Question 1

%Setting seed
rng(500)

%Clearing memory
clear all

%K-Means
%Importing exam images (I will keep the originals in case I need them later)
imgplane = imread('EECE5644_2019Fall_Homework4Questions_3096_colorPlane.jpg');
imgbird = imread('EECE5644_2019Fall_Homework4Questions_42049_colorBird.jpg');
imgserdar1 = imread('SO1.jpg');
imgserdar2 = imread('SO2.jpg');

%Converting colors and dimensions to a range between 0-1 and reducing
%dimensions
for i=1:3
    plane(:,:,i) = mat2gray(imgplane(:,:,i));
    bird(:,:,i) = mat2gray(imgbird(:,:,i));
    serdar1(:,:,i) = mat2gray(imgserdar1(:,:,i));
    serdar2(:,:,i) = mat2gray(imgserdar2(:,:,i));
end

%Dimension calculations for the plane and the bird are the same, the others
%are different
dimx = (0:1:size(plane,1)-1)/(size(plane,1)-1);
dimy = (0:1:size(plane,2)-1)/(size(plane,2)-1);
dimx1 = (0:1:size(serdar1,1)-1)/(size(serdar1,1)-1);
dimy1 = (0:1:size(serdar1,2)-1)/(size(serdar1,2)-1);
dimx2 = (0:1:size(serdar2,1)-1)/(size(serdar2,1)-1);
dimy2 = (0:1:size(serdar2,2)-1)/(size(serdar2,2)-1);

%Trying to condense dimensions into 0 to 1 in a single row
[dimX, dimY] = meshgrid(dimx, dimy);
[dimX1, dimY1] = meshgrid(dimx1, dimy1);
[dimX2, dimY2] = meshgrid(dimx2, dimy2);

plane(:,:,4) = dimX';
plane(:,:,5) = dimY';
bird(:,:,4) = dimX';
bird(:,:,5) = dimY';
serdar1(:,:,4) = dimX1';
serdar1(:,:,5) = dimY1';
serdar2(:,:,4) = dimX2';
serdar2(:,:,5) = dimY2';

A = 481*321;
B = 1937*2541;
C = 1877*2545;

%Reducing dimensions
plane = reshape(plane, [A,5]);
bird = reshape(bird, [A,5]);
serdar1 = reshape(serdar1, [B,5]);
serdar2 = reshape(serdar2, [C,5]);

%K-Means analysis
[idw2,A2] = kmeans(plane,2);
[idw3,A3] = kmeans(plane,3);
[idw4,A4] = kmeans(plane,4);
[idw5,A5] = kmeans(plane,5);
[idx2,B2] = kmeans(bird,2);
[idx3,B3] = kmeans(bird,3);
[idx4,B4] = kmeans(bird,4);
[idx5,B5] = kmeans(bird,5);
[idy2,C2] = kmeans(serdar1,2);
[idy3,C3] = kmeans(serdar1,3);
[idy4,C4] = kmeans(serdar1,4);
[idy5,C5] = kmeans(serdar1,5);
[idz2,D2] = kmeans(serdar2,2);
[idz3,D3] = kmeans(serdar2,3);
[idz4,D4] = kmeans(serdar2,4);
[idz5,D5] = kmeans(serdar2,5);

%Converting back to 3 dimensions
idW2 = reshape(idw2, [321,481]);
idW3 = reshape(idw3, [321,481]);
idW4 = reshape(idw4, [321,481]);
idW5 = reshape(idw5, [321,481]);
idX2 = reshape(idx2, [321,481]);
idX3 = reshape(idx3, [321,481]);
idX4 = reshape(idx4, [321,481]);
idX5 = reshape(idx5, [321,481]);
idY2 = reshape(idy2, [1937,2541]);
idY3 = reshape(idy3, [1937,2541]);
idY4 = reshape(idy4, [1937,2541]);
idY5 = reshape(idy5, [1937,2541]);
idZ2 = reshape(idz2, [1877,2545]);
idZ3 = reshape(idz3, [1877,2545]);
idZ4 = reshape(idz4, [1877,2545]);
idZ5 = reshape(idz5, [1877,2545]);

%Graphing Everything
figure(1)
imagesc(idW2)
figure(2)
imagesc(idW3)
figure(3)
imagesc(idW4)
figure(4)
imagesc(idW5)
figure(5)
imagesc(idX2)
figure(6)
imagesc(idX3)
figure(7)
imagesc(idX4)
figure(8)
imagesc(idX5)
figure(9)
imagesc(idY2)
figure(10)
imagesc(idY3)
figure(11)
imagesc(idY4)
figure(12)
imagesc(idY5)
figure(13)
imagesc(idZ2)
figure(14)
imagesc(idZ3)
figure(15)
imagesc(idZ4)
figure(16)
imagesc(idZ5)

%GMM+MAP Classifier
%I can't use the EMforGMM file since the matrix to be flipped becomes way
%too large (matlab can't do it this way)

%Starting with the plane
%2
gmm = fitgmdist(plane,2);
Sigma = gmm.Sigma;
m = gmm.mu';
p = gmm.ComponentProportion;
plane = plane';
%Loss values (0-1 for this error minimization)
lambda = [0 1;1 0]; 
%Threshold/decision determination
g1 = lambda(1,1)*evalGaussian(plane,m(:,1),Sigma(:,:,1))*p(1) + lambda(1,2)*evalGaussian(plane,m(:,2),Sigma(:,:,2))*p(2);
g2 = lambda(2,1)*evalGaussian(plane,m(:,1),Sigma(:,:,1))*p(1) + lambda(2,2)*evalGaussian(plane,m(:,2),Sigma(:,:,2))*p(2);
decision = zeros(1,A); 
for i = 1:A
    if g1(i)<g2(i)
        decision(i) = 1;
    elseif g2(i)<g1(i)
        decision(i) = 2;
end
end
%Reshaping and Graphing
decisioN = reshape(decision, [321,481]);
figure(17)
imagesc(decisioN)

%3
plane = plane';
gmm = fitgmdist(plane,3);
Sigma = gmm.Sigma;
m = gmm.mu';
p = gmm.ComponentProportion;
plane = plane';
%Loss values (0-1 for this error minimization)
lambda = [0 1 1;1 0 1;1 1 0]; 
%Threshold/decision determination
g1 = lambda(1,1)*evalGaussian(plane,m(:,1),Sigma(:,:,1))*p(1) + lambda(1,2)*evalGaussian(plane,m(:,2),Sigma(:,:,2))*p(2) + lambda(1,3)*evalGaussian(plane,m(:,3),Sigma(:,:,3))*p(3);
g2 = lambda(2,1)*evalGaussian(plane,m(:,1),Sigma(:,:,1))*p(1) + lambda(2,2)*evalGaussian(plane,m(:,2),Sigma(:,:,2))*p(2) + lambda(2,3)*evalGaussian(plane,m(:,3),Sigma(:,:,3))*p(3);
g3 = lambda(3,1)*evalGaussian(plane,m(:,1),Sigma(:,:,1))*p(1) + lambda(3,2)*evalGaussian(plane,m(:,2),Sigma(:,:,2))*p(2) + lambda(3,3)*evalGaussian(plane,m(:,3),Sigma(:,:,3))*p(3);
decision = zeros(1,A); 
for i = 1:A
    if g1(i)<g2(i) & g1(i)<g3(i)
        decision(i) = 1;
    elseif g2(i)<g1(i) & g2(i)<g3(i)
        decision(i) = 2;
    elseif g3(i)<g1(i) & g3(i)<g2(i)
        decision(i) = 3;
end
end
%Reshaping and Graphing
decisioN = reshape(decision, [321,481]);
figure(18)
imagesc(decisioN)

%4
plane = plane';
gmm = fitgmdist(plane,4);
Sigma = gmm.Sigma;
m = gmm.mu';
p = gmm.ComponentProportion;
plane = plane';
%Loss values (0-1 for this error minimization)
lambda = [0 1 1 1;1 0 1 1;1 1 0 1;1 1 1 0]; 
%Threshold/decision determination
g1 = lambda(1,1)*evalGaussian(plane,m(:,1),Sigma(:,:,1))*p(1) + lambda(1,2)*evalGaussian(plane,m(:,2),Sigma(:,:,2))*p(2) + lambda(1,3)*evalGaussian(plane,m(:,3),Sigma(:,:,3))*p(3) + lambda(1,4)*evalGaussian(plane,m(:,4),Sigma(:,:,4))*p(4);
g2 = lambda(2,1)*evalGaussian(plane,m(:,1),Sigma(:,:,1))*p(1) + lambda(2,2)*evalGaussian(plane,m(:,2),Sigma(:,:,2))*p(2) + lambda(2,3)*evalGaussian(plane,m(:,3),Sigma(:,:,3))*p(3) + lambda(2,4)*evalGaussian(plane,m(:,4),Sigma(:,:,4))*p(4);
g3 = lambda(3,1)*evalGaussian(plane,m(:,1),Sigma(:,:,1))*p(1) + lambda(3,2)*evalGaussian(plane,m(:,2),Sigma(:,:,2))*p(2) + lambda(3,3)*evalGaussian(plane,m(:,3),Sigma(:,:,3))*p(3) + lambda(3,4)*evalGaussian(plane,m(:,4),Sigma(:,:,4))*p(4);
g4 = lambda(4,1)*evalGaussian(plane,m(:,1),Sigma(:,:,1))*p(1) + lambda(4,2)*evalGaussian(plane,m(:,2),Sigma(:,:,2))*p(2) + lambda(4,3)*evalGaussian(plane,m(:,3),Sigma(:,:,3))*p(3) + lambda(4,4)*evalGaussian(plane,m(:,4),Sigma(:,:,4))*p(4);
decision = zeros(1,A); 
for i = 1:A
    if g1(i)<g2(i) & g1(i)<g3(i) & g1(i)<g4(i)
        decision(i) = 1;
    elseif g2(i)<g1(i) & g2(i)<g3(i) & g2(i)<g4(i)
        decision(i) = 2;
    elseif g3(i)<g1(i) & g3(i)<g2(i) & g3(i)<g4(i)
        decision(i) = 3;
    elseif g4(i)<g1(i) & g4(i)<g2(i) & g4(i)<g3(i)
        decision(i) = 4;
end
end
%Reshaping and Graphing
decisioN = reshape(decision, [321,481]);
figure(19)
imagesc(decisioN)

%5
plane = plane';
gmm = fitgmdist(plane,5);
Sigma = gmm.Sigma;
m = gmm.mu';
p = gmm.ComponentProportion;
plane = plane';
%Loss values (0-1 for this error minimization)
lambda = [0 1 1 1 1;1 0 1 1 1;1 1 0 1 1;1 1 1 0 1;1 1 1 1 0]; 
%Threshold/decision determination
g1 = lambda(1,1)*evalGaussian(plane,m(:,1),Sigma(:,:,1))*p(1) + lambda(1,2)*evalGaussian(plane,m(:,2),Sigma(:,:,2))*p(2) + lambda(1,3)*evalGaussian(plane,m(:,3),Sigma(:,:,3))*p(3) + lambda(1,4)*evalGaussian(plane,m(:,4),Sigma(:,:,4))*p(4) + lambda(1,5)*evalGaussian(plane,m(:,5),Sigma(:,:,5))*p(5);
g2 = lambda(2,1)*evalGaussian(plane,m(:,1),Sigma(:,:,1))*p(1) + lambda(2,2)*evalGaussian(plane,m(:,2),Sigma(:,:,2))*p(2) + lambda(2,3)*evalGaussian(plane,m(:,3),Sigma(:,:,3))*p(3) + lambda(2,4)*evalGaussian(plane,m(:,4),Sigma(:,:,4))*p(4) + lambda(2,5)*evalGaussian(plane,m(:,5),Sigma(:,:,5))*p(5);
g3 = lambda(3,1)*evalGaussian(plane,m(:,1),Sigma(:,:,1))*p(1) + lambda(3,2)*evalGaussian(plane,m(:,2),Sigma(:,:,2))*p(2) + lambda(3,3)*evalGaussian(plane,m(:,3),Sigma(:,:,3))*p(3) + lambda(3,4)*evalGaussian(plane,m(:,4),Sigma(:,:,4))*p(4) + lambda(3,5)*evalGaussian(plane,m(:,5),Sigma(:,:,5))*p(5);
g4 = lambda(4,1)*evalGaussian(plane,m(:,1),Sigma(:,:,1))*p(1) + lambda(4,2)*evalGaussian(plane,m(:,2),Sigma(:,:,2))*p(2) + lambda(4,3)*evalGaussian(plane,m(:,3),Sigma(:,:,3))*p(3) + lambda(4,4)*evalGaussian(plane,m(:,4),Sigma(:,:,4))*p(4) + lambda(4,5)*evalGaussian(plane,m(:,5),Sigma(:,:,5))*p(5);
g5 = lambda(5,1)*evalGaussian(plane,m(:,1),Sigma(:,:,1))*p(1) + lambda(5,2)*evalGaussian(plane,m(:,2),Sigma(:,:,2))*p(2) + lambda(5,3)*evalGaussian(plane,m(:,3),Sigma(:,:,3))*p(3) + lambda(5,4)*evalGaussian(plane,m(:,4),Sigma(:,:,4))*p(4) + lambda(5,5)*evalGaussian(plane,m(:,5),Sigma(:,:,5))*p(5);
decision = zeros(1,A); 
for i = 1:A
    if g1(i)<g2(i) & g1(i)<g3(i) & g1(i)<g4(i) & g1(i)<g5(i)
        decision(i) = 1;
    elseif g2(i)<g1(i) & g2(i)<g3(i) & g2(i)<g4(i) & g2(i)<g5(i)
        decision(i) = 2;
    elseif g3(i)<g1(i) & g3(i)<g2(i) & g3(i)<g4(i) & g3(i)<g5(i)
        decision(i) = 3;
    elseif g4(i)<g1(i) & g4(i)<g2(i) & g4(i)<g3(i) & g4(i)<g5(i)
        decision(i) = 4;
    elseif g5(i)<g1(i) & g5(i)<g2(i) & g5(i)<g3(i) & g5(i)<g4(i)
        decision(i) = 5;
end
end
%Reshaping and Graphing
decisioN = reshape(decision, [321,481]);
figure(20)
imagesc(decisioN)

%Then the bird
%2
gmm = fitgmdist(bird,2);
Sigma = gmm.Sigma;
m = gmm.mu';
p = gmm.ComponentProportion;
bird = bird';
%Loss values (0-1 for this error minimization)
lambda = [0 1;1 0]; 
%Threshold/decision determination
g1 = lambda(1,1)*evalGaussian(bird,m(:,1),Sigma(:,:,1))*p(1) + lambda(1,2)*evalGaussian(bird,m(:,2),Sigma(:,:,2))*p(2);
g2 = lambda(2,1)*evalGaussian(bird,m(:,1),Sigma(:,:,1))*p(1) + lambda(2,2)*evalGaussian(bird,m(:,2),Sigma(:,:,2))*p(2);
decision = zeros(1,A); 
for i = 1:A
    if g1(i)<g2(i)
        decision(i) = 1;
    elseif g2(i)<g1(i)
        decision(i) = 2;
end
end
%Reshaping and Graphing
decisioN = reshape(decision, [321,481]);
figure(21)
imagesc(decisioN)

%3
bird = bird';
gmm = fitgmdist(bird,3);
Sigma = gmm.Sigma;
m = gmm.mu';
p = gmm.ComponentProportion;
bird = bird';
%Loss values (0-1 for this error minimization)
lambda = [0 1 1;1 0 1;1 1 0]; 
%Threshold/decision determination
g1 = lambda(1,1)*evalGaussian(bird,m(:,1),Sigma(:,:,1))*p(1) + lambda(1,2)*evalGaussian(bird,m(:,2),Sigma(:,:,2))*p(2) + lambda(1,3)*evalGaussian(bird,m(:,3),Sigma(:,:,3))*p(3);
g2 = lambda(2,1)*evalGaussian(bird,m(:,1),Sigma(:,:,1))*p(1) + lambda(2,2)*evalGaussian(bird,m(:,2),Sigma(:,:,2))*p(2) + lambda(2,3)*evalGaussian(bird,m(:,3),Sigma(:,:,3))*p(3);
g3 = lambda(3,1)*evalGaussian(bird,m(:,1),Sigma(:,:,1))*p(1) + lambda(3,2)*evalGaussian(bird,m(:,2),Sigma(:,:,2))*p(2) + lambda(3,3)*evalGaussian(bird,m(:,3),Sigma(:,:,3))*p(3);
decision = zeros(1,A); 
for i = 1:A
    if g1(i)<g2(i) & g1(i)<g3(i)
        decision(i) = 1;
    elseif g2(i)<g1(i) & g2(i)<g3(i)
        decision(i) = 2;
    elseif g3(i)<g1(i) & g3(i)<g2(i)
        decision(i) = 3;
end
end
%Reshaping and Graphing
decisioN = reshape(decision, [321,481]);
figure(22)
imagesc(decisioN)

%4
bird = bird';
gmm = fitgmdist(bird,4);
Sigma = gmm.Sigma;
m = gmm.mu';
p = gmm.ComponentProportion;
bird = bird';
%Loss values (0-1 for this error minimization)
lambda = [0 1 1 1;1 0 1 1;1 1 0 1;1 1 1 0]; 
%Threshold/decision determination
g1 = lambda(1,1)*evalGaussian(bird,m(:,1),Sigma(:,:,1))*p(1) + lambda(1,2)*evalGaussian(bird,m(:,2),Sigma(:,:,2))*p(2) + lambda(1,3)*evalGaussian(bird,m(:,3),Sigma(:,:,3))*p(3) + lambda(1,4)*evalGaussian(bird,m(:,4),Sigma(:,:,4))*p(4);
g2 = lambda(2,1)*evalGaussian(bird,m(:,1),Sigma(:,:,1))*p(1) + lambda(2,2)*evalGaussian(bird,m(:,2),Sigma(:,:,2))*p(2) + lambda(2,3)*evalGaussian(bird,m(:,3),Sigma(:,:,3))*p(3) + lambda(2,4)*evalGaussian(bird,m(:,4),Sigma(:,:,4))*p(4);
g3 = lambda(3,1)*evalGaussian(bird,m(:,1),Sigma(:,:,1))*p(1) + lambda(3,2)*evalGaussian(bird,m(:,2),Sigma(:,:,2))*p(2) + lambda(3,3)*evalGaussian(bird,m(:,3),Sigma(:,:,3))*p(3) + lambda(3,4)*evalGaussian(bird,m(:,4),Sigma(:,:,4))*p(4);
g4 = lambda(4,1)*evalGaussian(bird,m(:,1),Sigma(:,:,1))*p(1) + lambda(4,2)*evalGaussian(bird,m(:,2),Sigma(:,:,2))*p(2) + lambda(4,3)*evalGaussian(bird,m(:,3),Sigma(:,:,3))*p(3) + lambda(4,4)*evalGaussian(bird,m(:,4),Sigma(:,:,4))*p(4);
decision = zeros(1,A); 
for i = 1:A
    if g1(i)<g2(i) & g1(i)<g3(i) & g1(i)<g4(i)
        decision(i) = 1;
    elseif g2(i)<g1(i) & g2(i)<g3(i) & g2(i)<g4(i)
        decision(i) = 2;
    elseif g3(i)<g1(i) & g3(i)<g2(i) & g3(i)<g4(i)
        decision(i) = 3;
    elseif g4(i)<g1(i) & g4(i)<g2(i) & g4(i)<g3(i)
        decision(i) = 4;
end
end
%Reshaping and Graphing
decisioN = reshape(decision, [321,481]);
figure(23)
imagesc(decisioN)

%5
bird = bird';
gmm = fitgmdist(bird,5);
Sigma = gmm.Sigma;
m = gmm.mu';
p = gmm.ComponentProportion;
bird = bird';
%Loss values (0-1 for this error minimization)
lambda = [0 1 1 1 1;1 0 1 1 1;1 1 0 1 1;1 1 1 0 1;1 1 1 1 0]; 
%Threshold/decision determination
g1 = lambda(1,1)*evalGaussian(bird,m(:,1),Sigma(:,:,1))*p(1) + lambda(1,2)*evalGaussian(bird,m(:,2),Sigma(:,:,2))*p(2) + lambda(1,3)*evalGaussian(bird,m(:,3),Sigma(:,:,3))*p(3) + lambda(1,4)*evalGaussian(bird,m(:,4),Sigma(:,:,4))*p(4) + lambda(1,5)*evalGaussian(bird,m(:,5),Sigma(:,:,5))*p(5);
g2 = lambda(2,1)*evalGaussian(bird,m(:,1),Sigma(:,:,1))*p(1) + lambda(2,2)*evalGaussian(bird,m(:,2),Sigma(:,:,2))*p(2) + lambda(2,3)*evalGaussian(bird,m(:,3),Sigma(:,:,3))*p(3) + lambda(2,4)*evalGaussian(bird,m(:,4),Sigma(:,:,4))*p(4) + lambda(2,5)*evalGaussian(bird,m(:,5),Sigma(:,:,5))*p(5);
g3 = lambda(3,1)*evalGaussian(bird,m(:,1),Sigma(:,:,1))*p(1) + lambda(3,2)*evalGaussian(bird,m(:,2),Sigma(:,:,2))*p(2) + lambda(3,3)*evalGaussian(bird,m(:,3),Sigma(:,:,3))*p(3) + lambda(3,4)*evalGaussian(bird,m(:,4),Sigma(:,:,4))*p(4) + lambda(3,5)*evalGaussian(bird,m(:,5),Sigma(:,:,5))*p(5);
g4 = lambda(4,1)*evalGaussian(bird,m(:,1),Sigma(:,:,1))*p(1) + lambda(4,2)*evalGaussian(bird,m(:,2),Sigma(:,:,2))*p(2) + lambda(4,3)*evalGaussian(bird,m(:,3),Sigma(:,:,3))*p(3) + lambda(4,4)*evalGaussian(bird,m(:,4),Sigma(:,:,4))*p(4) + lambda(4,5)*evalGaussian(bird,m(:,5),Sigma(:,:,5))*p(5);
g5 = lambda(5,1)*evalGaussian(bird,m(:,1),Sigma(:,:,1))*p(1) + lambda(5,2)*evalGaussian(bird,m(:,2),Sigma(:,:,2))*p(2) + lambda(5,3)*evalGaussian(bird,m(:,3),Sigma(:,:,3))*p(3) + lambda(5,4)*evalGaussian(bird,m(:,4),Sigma(:,:,4))*p(4) + lambda(5,5)*evalGaussian(bird,m(:,5),Sigma(:,:,5))*p(5);
decision = zeros(1,A); 
for i = 1:A
    if g1(i)<g2(i) & g1(i)<g3(i) & g1(i)<g4(i) & g1(i)<g5(i)
        decision(i) = 1;
    elseif g2(i)<g1(i) & g2(i)<g3(i) & g2(i)<g4(i) & g2(i)<g5(i)
        decision(i) = 2;
    elseif g3(i)<g1(i) & g3(i)<g2(i) & g3(i)<g4(i) & g3(i)<g5(i)
        decision(i) = 3;
    elseif g4(i)<g1(i) & g4(i)<g2(i) & g4(i)<g3(i) & g4(i)<g5(i)
        decision(i) = 4;
    elseif g5(i)<g1(i) & g5(i)<g2(i) & g5(i)<g3(i) & g5(i)<g4(i)
        decision(i) = 5;
end
end
%Reshaping and Graphing
decisioN = reshape(decision, [321,481]);
figure(24)
imagesc(decisioN)

%Then serdar1
%2
gmm = fitgmdist(serdar1,2);
Sigma = gmm.Sigma;
m = gmm.mu';
p = gmm.ComponentProportion;
serdar1 = serdar1';
%Loss values (0-1 for this error minimization)
lambda = [0 1;1 0]; 
%Threshold/decision determination
g1 = lambda(1,1)*evalGaussian(serdar1,m(:,1),Sigma(:,:,1))*p(1) + lambda(1,2)*evalGaussian(serdar1,m(:,2),Sigma(:,:,2))*p(2);
g2 = lambda(2,1)*evalGaussian(serdar1,m(:,1),Sigma(:,:,1))*p(1) + lambda(2,2)*evalGaussian(serdar1,m(:,2),Sigma(:,:,2))*p(2);
decision = zeros(1,B); 
for i = 1:B
    if g1(i)<g2(i)
        decision(i) = 1;
    elseif g2(i)<g1(i)
        decision(i) = 2;
end
end
%Reshaping and Graphing
decisioN = reshape(decision, [1937,2541]);
figure(25)
imagesc(decisioN)

%3
serdar1 = serdar1';
gmm = fitgmdist(serdar1,3);
Sigma = gmm.Sigma;
m = gmm.mu';
p = gmm.ComponentProportion;
serdar1 = serdar1';
%Loss values (0-1 for this error minimization)
lambda = [0 1 1;1 0 1;1 1 0]; 
%Threshold/decision determination
g1 = lambda(1,1)*evalGaussian(serdar1,m(:,1),Sigma(:,:,1))*p(1) + lambda(1,2)*evalGaussian(serdar1,m(:,2),Sigma(:,:,2))*p(2) + lambda(1,3)*evalGaussian(serdar1,m(:,3),Sigma(:,:,3))*p(3);
g2 = lambda(2,1)*evalGaussian(serdar1,m(:,1),Sigma(:,:,1))*p(1) + lambda(2,2)*evalGaussian(serdar1,m(:,2),Sigma(:,:,2))*p(2) + lambda(2,3)*evalGaussian(serdar1,m(:,3),Sigma(:,:,3))*p(3);
g3 = lambda(3,1)*evalGaussian(serdar1,m(:,1),Sigma(:,:,1))*p(1) + lambda(3,2)*evalGaussian(serdar1,m(:,2),Sigma(:,:,2))*p(2) + lambda(3,3)*evalGaussian(serdar1,m(:,3),Sigma(:,:,3))*p(3);
decision = zeros(1,B); 
for i = 1:B
    if g1(i)<g2(i) & g1(i)<g3(i)
        decision(i) = 1;
    elseif g2(i)<g1(i) & g2(i)<g3(i)
        decision(i) = 2;
    elseif g3(i)<g1(i) & g3(i)<g2(i)
        decision(i) = 3;
end
end
%Reshaping and Graphing
decisioN = reshape(decision, [1937,2541]);
figure(26)
imagesc(decisioN)

%4
serdar1 = serdar1';
gmm = fitgmdist(serdar1,4);
Sigma = gmm.Sigma;
m = gmm.mu';
p = gmm.ComponentProportion;
serdar1 = serdar1';
%Loss values (0-1 for this error minimization)
lambda = [0 1 1 1;1 0 1 1;1 1 0 1;1 1 1 0]; 
%Threshold/decision determination
g1 = lambda(1,1)*evalGaussian(serdar1,m(:,1),Sigma(:,:,1))*p(1) + lambda(1,2)*evalGaussian(serdar1,m(:,2),Sigma(:,:,2))*p(2) + lambda(1,3)*evalGaussian(serdar1,m(:,3),Sigma(:,:,3))*p(3) + lambda(1,4)*evalGaussian(serdar1,m(:,4),Sigma(:,:,4))*p(4);
g2 = lambda(2,1)*evalGaussian(serdar1,m(:,1),Sigma(:,:,1))*p(1) + lambda(2,2)*evalGaussian(serdar1,m(:,2),Sigma(:,:,2))*p(2) + lambda(2,3)*evalGaussian(serdar1,m(:,3),Sigma(:,:,3))*p(3) + lambda(2,4)*evalGaussian(serdar1,m(:,4),Sigma(:,:,4))*p(4);
g3 = lambda(3,1)*evalGaussian(serdar1,m(:,1),Sigma(:,:,1))*p(1) + lambda(3,2)*evalGaussian(serdar1,m(:,2),Sigma(:,:,2))*p(2) + lambda(3,3)*evalGaussian(serdar1,m(:,3),Sigma(:,:,3))*p(3) + lambda(3,4)*evalGaussian(serdar1,m(:,4),Sigma(:,:,4))*p(4);
g4 = lambda(4,1)*evalGaussian(serdar1,m(:,1),Sigma(:,:,1))*p(1) + lambda(4,2)*evalGaussian(serdar1,m(:,2),Sigma(:,:,2))*p(2) + lambda(4,3)*evalGaussian(serdar1,m(:,3),Sigma(:,:,3))*p(3) + lambda(4,4)*evalGaussian(serdar1,m(:,4),Sigma(:,:,4))*p(4);
decision = zeros(1,B); 
for i = 1:B
    if g1(i)<g2(i) & g1(i)<g3(i) & g1(i)<g4(i)
        decision(i) = 1;
    elseif g2(i)<g1(i) & g2(i)<g3(i) & g2(i)<g4(i)
        decision(i) = 2;
    elseif g3(i)<g1(i) & g3(i)<g2(i) & g3(i)<g4(i)
        decision(i) = 3;
    elseif g4(i)<g1(i) & g4(i)<g2(i) & g4(i)<g3(i)
        decision(i) = 4;
end
end
%Reshaping and Graphing
decisioN = reshape(decision, [1937,2541]);
figure(27)
imagesc(decisioN)

%5
serdar1 = serdar1';
gmm = fitgmdist(serdar1,5);
Sigma = gmm.Sigma;
m = gmm.mu';
p = gmm.ComponentProportion;
serdar1 = serdar1';
%Loss values (0-1 for this error minimization)
lambda = [0 1 1 1 1;1 0 1 1 1;1 1 0 1 1;1 1 1 0 1;1 1 1 1 0]; 
%Threshold/decision determination
g1 = lambda(1,1)*evalGaussian(serdar1,m(:,1),Sigma(:,:,1))*p(1) + lambda(1,2)*evalGaussian(serdar1,m(:,2),Sigma(:,:,2))*p(2) + lambda(1,3)*evalGaussian(serdar1,m(:,3),Sigma(:,:,3))*p(3) + lambda(1,4)*evalGaussian(serdar1,m(:,4),Sigma(:,:,4))*p(4) + lambda(1,5)*evalGaussian(serdar1,m(:,5),Sigma(:,:,5))*p(5);
g2 = lambda(2,1)*evalGaussian(serdar1,m(:,1),Sigma(:,:,1))*p(1) + lambda(2,2)*evalGaussian(serdar1,m(:,2),Sigma(:,:,2))*p(2) + lambda(2,3)*evalGaussian(serdar1,m(:,3),Sigma(:,:,3))*p(3) + lambda(2,4)*evalGaussian(serdar1,m(:,4),Sigma(:,:,4))*p(4) + lambda(2,5)*evalGaussian(serdar1,m(:,5),Sigma(:,:,5))*p(5);
g3 = lambda(3,1)*evalGaussian(serdar1,m(:,1),Sigma(:,:,1))*p(1) + lambda(3,2)*evalGaussian(serdar1,m(:,2),Sigma(:,:,2))*p(2) + lambda(3,3)*evalGaussian(serdar1,m(:,3),Sigma(:,:,3))*p(3) + lambda(3,4)*evalGaussian(serdar1,m(:,4),Sigma(:,:,4))*p(4) + lambda(3,5)*evalGaussian(serdar1,m(:,5),Sigma(:,:,5))*p(5);
g4 = lambda(4,1)*evalGaussian(serdar1,m(:,1),Sigma(:,:,1))*p(1) + lambda(4,2)*evalGaussian(serdar1,m(:,2),Sigma(:,:,2))*p(2) + lambda(4,3)*evalGaussian(serdar1,m(:,3),Sigma(:,:,3))*p(3) + lambda(4,4)*evalGaussian(serdar1,m(:,4),Sigma(:,:,4))*p(4) + lambda(4,5)*evalGaussian(serdar1,m(:,5),Sigma(:,:,5))*p(5);
g5 = lambda(5,1)*evalGaussian(serdar1,m(:,1),Sigma(:,:,1))*p(1) + lambda(5,2)*evalGaussian(serdar1,m(:,2),Sigma(:,:,2))*p(2) + lambda(5,3)*evalGaussian(serdar1,m(:,3),Sigma(:,:,3))*p(3) + lambda(5,4)*evalGaussian(serdar1,m(:,4),Sigma(:,:,4))*p(4) + lambda(5,5)*evalGaussian(serdar1,m(:,5),Sigma(:,:,5))*p(5);
decision = zeros(1,B); 
for i = 1:B
    if g1(i)<g2(i) & g1(i)<g3(i) & g1(i)<g4(i) & g1(i)<g5(i)
        decision(i) = 1;
    elseif g2(i)<g1(i) & g2(i)<g3(i) & g2(i)<g4(i) & g2(i)<g5(i)
        decision(i) = 2;
    elseif g3(i)<g1(i) & g3(i)<g2(i) & g3(i)<g4(i) & g3(i)<g5(i)
        decision(i) = 3;
    elseif g4(i)<g1(i) & g4(i)<g2(i) & g4(i)<g3(i) & g4(i)<g5(i)
        decision(i) = 4;
    elseif g5(i)<g1(i) & g5(i)<g2(i) & g5(i)<g3(i) & g5(i)<g4(i)
        decision(i) = 5;
end
end
%Reshaping and Graphing
decisioN = reshape(decision, [1937,2541]);
figure(28)
imagesc(decisioN)

%Finally serdar2
%2
gmm = fitgmdist(serdar2,2);
Sigma = gmm.Sigma;
m = gmm.mu';
p = gmm.ComponentProportion;
serdar2 = serdar2';
%Loss values (0-1 for this error minimization)
lambda = [0 1;1 0]; 
%Threshold/decision determination
g1 = lambda(1,1)*evalGaussian(serdar2,m(:,1),Sigma(:,:,1))*p(1) + lambda(1,2)*evalGaussian(serdar2,m(:,2),Sigma(:,:,2))*p(2);
g2 = lambda(2,1)*evalGaussian(serdar2,m(:,1),Sigma(:,:,1))*p(1) + lambda(2,2)*evalGaussian(serdar2,m(:,2),Sigma(:,:,2))*p(2);
decision = zeros(1,C); 
for i = 1:C
    if g1(i)<g2(i)
        decision(i) = 1;
    elseif g2(i)<g1(i)
        decision(i) = 2;
end
end
%Reshaping and Graphing
decisioN = reshape(decision, [1877,2545]);
figure(29)
imagesc(decisioN)

%3
serdar2 = serdar2';
gmm = fitgmdist(serdar2,3);
Sigma = gmm.Sigma;
m = gmm.mu';
p = gmm.ComponentProportion;
serdar2 = serdar2';
%Loss values (0-1 for this error minimization)
lambda = [0 1 1;1 0 1;1 1 0]; 
%Threshold/decision determination
g1 = lambda(1,1)*evalGaussian(serdar2,m(:,1),Sigma(:,:,1))*p(1) + lambda(1,2)*evalGaussian(serdar2,m(:,2),Sigma(:,:,2))*p(2) + lambda(1,3)*evalGaussian(serdar2,m(:,3),Sigma(:,:,3))*p(3);
g2 = lambda(2,1)*evalGaussian(serdar2,m(:,1),Sigma(:,:,1))*p(1) + lambda(2,2)*evalGaussian(serdar2,m(:,2),Sigma(:,:,2))*p(2) + lambda(2,3)*evalGaussian(serdar2,m(:,3),Sigma(:,:,3))*p(3);
g3 = lambda(3,1)*evalGaussian(serdar2,m(:,1),Sigma(:,:,1))*p(1) + lambda(3,2)*evalGaussian(serdar2,m(:,2),Sigma(:,:,2))*p(2) + lambda(3,3)*evalGaussian(serdar2,m(:,3),Sigma(:,:,3))*p(3);
decision = zeros(1,C); 
for i = 1:C
    if g1(i)<g2(i) & g1(i)<g3(i)
        decision(i) = 1;
    elseif g2(i)<g1(i) & g2(i)<g3(i)
        decision(i) = 2;
    elseif g3(i)<g1(i) & g3(i)<g2(i)
        decision(i) = 3;
end
end
%Reshaping and Graphing
decisioN = reshape(decision, [1877,2545]);
figure(30)
imagesc(decisioN)

%4
serdar2 = serdar2';
gmm = fitgmdist(serdar2,4);
Sigma = gmm.Sigma;
m = gmm.mu';
p = gmm.ComponentProportion;
serdar2 = serdar2';
%Loss values (0-1 for this error minimization)
lambda = [0 1 1 1;1 0 1 1;1 1 0 1;1 1 1 0]; 
%Threshold/decision determination
g1 = lambda(1,1)*evalGaussian(serdar2,m(:,1),Sigma(:,:,1))*p(1) + lambda(1,2)*evalGaussian(serdar2,m(:,2),Sigma(:,:,2))*p(2) + lambda(1,3)*evalGaussian(serdar2,m(:,3),Sigma(:,:,3))*p(3) + lambda(1,4)*evalGaussian(serdar2,m(:,4),Sigma(:,:,4))*p(4);
g2 = lambda(2,1)*evalGaussian(serdar2,m(:,1),Sigma(:,:,1))*p(1) + lambda(2,2)*evalGaussian(serdar2,m(:,2),Sigma(:,:,2))*p(2) + lambda(2,3)*evalGaussian(serdar2,m(:,3),Sigma(:,:,3))*p(3) + lambda(2,4)*evalGaussian(serdar2,m(:,4),Sigma(:,:,4))*p(4);
g3 = lambda(3,1)*evalGaussian(serdar2,m(:,1),Sigma(:,:,1))*p(1) + lambda(3,2)*evalGaussian(serdar2,m(:,2),Sigma(:,:,2))*p(2) + lambda(3,3)*evalGaussian(serdar2,m(:,3),Sigma(:,:,3))*p(3) + lambda(3,4)*evalGaussian(serdar2,m(:,4),Sigma(:,:,4))*p(4);
g4 = lambda(4,1)*evalGaussian(serdar2,m(:,1),Sigma(:,:,1))*p(1) + lambda(4,2)*evalGaussian(serdar2,m(:,2),Sigma(:,:,2))*p(2) + lambda(4,3)*evalGaussian(serdar2,m(:,3),Sigma(:,:,3))*p(3) + lambda(4,4)*evalGaussian(serdar2,m(:,4),Sigma(:,:,4))*p(4);
decision = zeros(1,C); 
for i = 1:C
    if g1(i)<g2(i) & g1(i)<g3(i) & g1(i)<g4(i)
        decision(i) = 1;
    elseif g2(i)<g1(i) & g2(i)<g3(i) & g2(i)<g4(i)
        decision(i) = 2;
    elseif g3(i)<g1(i) & g3(i)<g2(i) & g3(i)<g4(i)
        decision(i) = 3;
    elseif g4(i)<g1(i) & g4(i)<g2(i) & g4(i)<g3(i)
        decision(i) = 4;
end
end
%Reshaping and Graphing
decisioN = reshape(decision, [1877,2545]);
figure(31)
imagesc(decisioN)

%5
serdar2 = serdar2';
gmm = fitgmdist(serdar2,5);
Sigma = gmm.Sigma;
m = gmm.mu';
p = gmm.ComponentProportion;
serdar2 = serdar2';
%Loss values (0-1 for this error minimization)
lambda = [0 1 1 1 1;1 0 1 1 1;1 1 0 1 1;1 1 1 0 1;1 1 1 1 0]; 
%Threshold/decision determination
g1 = lambda(1,1)*evalGaussian(serdar2,m(:,1),Sigma(:,:,1))*p(1) + lambda(1,2)*evalGaussian(serdar2,m(:,2),Sigma(:,:,2))*p(2) + lambda(1,3)*evalGaussian(serdar2,m(:,3),Sigma(:,:,3))*p(3) + lambda(1,4)*evalGaussian(serdar2,m(:,4),Sigma(:,:,4))*p(4) + lambda(1,5)*evalGaussian(serdar2,m(:,5),Sigma(:,:,5))*p(5);
g2 = lambda(2,1)*evalGaussian(serdar2,m(:,1),Sigma(:,:,1))*p(1) + lambda(2,2)*evalGaussian(serdar2,m(:,2),Sigma(:,:,2))*p(2) + lambda(2,3)*evalGaussian(serdar2,m(:,3),Sigma(:,:,3))*p(3) + lambda(2,4)*evalGaussian(serdar2,m(:,4),Sigma(:,:,4))*p(4) + lambda(2,5)*evalGaussian(serdar2,m(:,5),Sigma(:,:,5))*p(5);
g3 = lambda(3,1)*evalGaussian(serdar2,m(:,1),Sigma(:,:,1))*p(1) + lambda(3,2)*evalGaussian(serdar2,m(:,2),Sigma(:,:,2))*p(2) + lambda(3,3)*evalGaussian(serdar2,m(:,3),Sigma(:,:,3))*p(3) + lambda(3,4)*evalGaussian(serdar2,m(:,4),Sigma(:,:,4))*p(4) + lambda(3,5)*evalGaussian(serdar2,m(:,5),Sigma(:,:,5))*p(5);
g4 = lambda(4,1)*evalGaussian(serdar2,m(:,1),Sigma(:,:,1))*p(1) + lambda(4,2)*evalGaussian(serdar2,m(:,2),Sigma(:,:,2))*p(2) + lambda(4,3)*evalGaussian(serdar2,m(:,3),Sigma(:,:,3))*p(3) + lambda(4,4)*evalGaussian(serdar2,m(:,4),Sigma(:,:,4))*p(4) + lambda(4,5)*evalGaussian(serdar2,m(:,5),Sigma(:,:,5))*p(5);
g5 = lambda(5,1)*evalGaussian(serdar2,m(:,1),Sigma(:,:,1))*p(1) + lambda(5,2)*evalGaussian(serdar2,m(:,2),Sigma(:,:,2))*p(2) + lambda(5,3)*evalGaussian(serdar2,m(:,3),Sigma(:,:,3))*p(3) + lambda(5,4)*evalGaussian(serdar2,m(:,4),Sigma(:,:,4))*p(4) + lambda(5,5)*evalGaussian(serdar2,m(:,5),Sigma(:,:,5))*p(5);
decision = zeros(1,C); 
for i = 1:C
    if g1(i)<g2(i) & g1(i)<g3(i) & g1(i)<g4(i) & g1(i)<g5(i)
        decision(i) = 1;
    elseif g2(i)<g1(i) & g2(i)<g3(i) & g2(i)<g4(i) & g2(i)<g5(i)
        decision(i) = 2;
    elseif g3(i)<g1(i) & g3(i)<g2(i) & g3(i)<g4(i) & g3(i)<g5(i)
        decision(i) = 3;
    elseif g4(i)<g1(i) & g4(i)<g2(i) & g4(i)<g3(i) & g4(i)<g5(i)
        decision(i) = 4;
    elseif g5(i)<g1(i) & g5(i)<g2(i) & g5(i)<g3(i) & g5(i)<g4(i)
        decision(i) = 5;
end
end
%Reshaping and Graphing
decisioN = reshape(decision, [1877,2545]);
figure(32)
imagesc(decisioN)

%%%
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end