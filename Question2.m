%Question 2

%Setting seed
rng(50500)

%Start SVM
close all, clear all,
N=1000; n = 2; K=10;
mu(:,1) = [0;0];
Sigma(:,:,1) = [1 0;0 1]; 
%Class priors for labels -1 and +1 respectively
p = [0.35,0.65];
%Generate samples (training)
label = rand(1,N) >= p(1); l = 2*(label-0.5);
%Number of samples from each class
Nc = [length(find(label==0)),length(find(label==1))];
%Reserve space
x = zeros(n,N); 
%Draw samples from each class pdf
%Draw from radius 
rad(:,label==1) = rand(1,length(x(:,label==1)))+2;
ang(:,label==1) = rand(1,length(x(:,label==1)))*2*pi-pi;
[x(1,:),x(2,:)] = pol2cart(ang,rad);
for lbl = 0:0
    x(:,label==lbl) = randGaussian(Nc(lbl+1),mu(:,lbl+1),Sigma(:,:,lbl+1));
end

figure(1)
plot(x(1,label==0),x(2,label==0),'ob'), hold on,
plot(x(1,label==1),x(2,label==1),'+r'),  axis equal,
legend('Class -1','Class +1'), 
title('Training Data with Real Population/Class Labels'),
xlabel('x_1'), ylabel('x_2'), 

%Generate samples (validation)
labelv = rand(1,N) >= p(1); lv = 2*(labelv-0.5);
%Number of samples from each class
Ncv = [length(find(labelv==0)),length(find(labelv==1))]; 
%Reserve space
xv = zeros(n,N); 
%Draw samples from each class pdf
%Draw from radius 
radv(:,labelv==1) = rand(1,length(xv(:,labelv==1)))+2;
angv(:,labelv==1) = rand(1,length(xv(:,labelv==1)))*2*pi-pi;
[xv(1,:),xv(2,:)] = pol2cart(angv,radv);
for lbl = 0:0
    xv(:,labelv==lbl) = randGaussian(Ncv(lbl+1),mu(:,lbl+1),Sigma(:,:,lbl+1));
end

figure(2)
plot(xv(1,labelv==0),xv(2,labelv==0),'ob'), hold on,
plot(xv(1,labelv==1),xv(2,labelv==1),'+r'),  axis equal,
legend('Class -1','Class +1'), 
title('Validation Data with Real Population/Class Labels'),
xlabel('x_1'), ylabel('x_2'), 

%Train a Linear kernel SVM with cross-validation
%to select hyperparameters that minimize probability 
%of error (i.e. maximize accuracy; 0-1 loss scenario)
dummy = ceil(linspace(0,N,K+1));
for k = 1:K, indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)]; end,
CList = 10.^linspace(-3,7,11);
for CCounter = 1:length(CList)
    [CCounter,length(CList)],
    C = CList(CCounter);
    for k = 1:K
        indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
        %Using fold k as the validation set
        xValidate = x(:,indValidate); 
        lValidate = l(indValidate);
        if k == 1
            indTrain = [indPartitionLimits(k,2)+1:N];
        elseif k == K
            indTrain = [1:indPartitionLimits(k,1)-1];
        else
            indTrain = [1:indPartitionLimits(k-1,2),indPartitionLimits(k+1,1):N];
        end
        %Using all other folds as the training set
        xTrain = x(:,indTrain); lTrain = l(indTrain);
        SVMk = fitcsvm(xTrain',lTrain,'BoxConstraint',C,'KernelFunction','linear');
        %Labels of validation data using the trained SVM
        dValidate = SVMk.predict(xValidate')'; 
        indCORRECT = find(lValidate.*dValidate == 1); 
        Ncorrect(k)=length(indCORRECT);
    end 
    PCorrect(CCounter)= sum(Ncorrect)/N; 
end 
figure(3), 
plot(log10(CList),PCorrect,'.',log10(CList),PCorrect,'-'),
xlabel('log_{10} C'),ylabel('K-fold Validation Accuracy Estimate'),
title('Linear-SVM Cross-Val Accuracy Estimate'), 
[dummy,indi] = max(PCorrect(:)); [indBestC, indBestSigma] = ind2sub(size(PCorrect),indi);
CBest= CList(indBestC); 
SVMBest = fitcsvm(x',l','BoxConstraint',CBest,'KernelFunction','linear');
%Labels of training data using the trained SVM
d = SVMBest.predict(x')'; 
%Find training samples that are incorrectly classified by the trained SVM
indINCORRECT = find(l.*d == -1); 
%Find training samples that are correctly classified by the trained SVM
indCORRECT = find(l.*d == 1); 
ind00 = find(d==-1 & l==-1); 
ind10 = find(d==1 & l==-1); 
ind01 = find(d==-1 & l==1);
ind11 = find(d==1 & l==1);
figure(4), 
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,
pTrainingError = length(indINCORRECT)/N, 
%Empirical estimate of training error probability
%Grid search (unnecessary for linear kernels)
Nx = 10010; Ny = 9900; xGrid = linspace(-10,10,Nx); yGrid = linspace(-10,10,Ny);
[h,v] = meshgrid(xGrid,yGrid); dGrid = SVMBest.predict([h(:),v(:)]); zGrid = reshape(dGrid,Ny,Nx);
figure(4) , contour(xGrid,yGrid,zGrid,10); xlabel('x1'), ylabel('x2'), axis equal,
legend('Wrong decisions for data from Class -1','Correct decisions for data from Class +1','Equilevel contours of the discriminant function' ), 
title('Training Data with Decisions'),
hold off

%Independent Test Sample
%Labels of independent data
dindep = SVMBest.predict(xv')'; 
indINCORRECTv = find(lv.*dindep == -1); 
indCORRECTv = find(lv.*dindep == 1);
pTrainingErrorv = length(indINCORRECTv)/N,
ind00 = find(dindep==-1 & lv==-1); 
ind10 = find(dindep==1 & lv==-1); 
ind01 = find(dindep==-1 & lv==1);
ind11 = find(dindep==1 & lv==1);
figure(5),
plot(xv(1,ind00),xv(2,ind00),'og'); hold on,
plot(xv(1,ind10),xv(2,ind10),'or'); hold on,
plot(xv(1,ind01),xv(2,ind01),'+r'); hold on,
plot(xv(1,ind11),xv(2,ind11),'+g'); hold on,
axis equal,
%Grid search (unnecessary for linear kernels)
Nx = 10010; Ny = 9900; xGrid = linspace(-10,10,Nx); yGrid = linspace(-10,10,Ny);
[h,v] = meshgrid(xGrid,yGrid); dGrid = SVMBest.predict([h(:),v(:)]); zGrid = reshape(dGrid,Ny,Nx);
figure(5),  contour(xGrid,yGrid,zGrid,10); xlabel('x1'), ylabel('x2'), axis equal,
legend('Wrong decisions for data from Class -1','Correct decisions for data from Class +1','Equilevel contours of the discriminant function' ), 
title('Validation Data with Decisions'),
hold off

%Train a Gaussian kernel SVM with cross-validation
%to select hyperparameters that minimize probability 
%of error (i.e. maximize accuracy; 0-1 loss scenario)
dummy = ceil(linspace(0,N,K+1));
for k = 1:K, indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)]; end,
CList = 10.^linspace(-1,9,11); sigmaList = 10.^linspace(-2,3,13);
for sigmaCounter = 1:length(sigmaList)
    [sigmaCounter,length(sigmaList)],
    sigma = sigmaList(sigmaCounter);
    for CCounter = 1:length(CList)
        C = CList(CCounter);
        for k = 1:K
            indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
            %Using fold k as validation set
            xValidate = x(:,indValidate); 
            lValidate = l(indValidate);  
            if k == 1
                indTrain = [indPartitionLimits(k,2)+1:N];
            elseif k == K
                indTrain = [1:indPartitionLimits(k,1)-1];
            else
                indTrain = [1:indPartitionLimits(k-1,2),indPartitionLimits(k+1,1):N];
            end
            %Using all other folds as training set
            xTrain = x(:,indTrain); lTrain = l(indTrain);
            SVMk = fitcsvm(xTrain',lTrain,'BoxConstraint',C,'KernelFunction','gaussian','KernelScale',sigma);
            %Labels of validation data using the trained SVM
            dValidate = SVMk.predict(xValidate')'; 
            indCORRECT = find(lValidate.*dValidate == 1); 
            Ncorrect(k)=length(indCORRECT);
        end 
        PCorrect(CCounter,sigmaCounter)= sum(Ncorrect)/N;
    end 
end
figure(6),
contour(log10(CList),log10(sigmaList),PCorrect',20); xlabel('log_{10} C'), ylabel('log_{10} sigma'),
title('Gaussian-SVM Cross-Val Accuracy Estimate'), axis equal,
[dummy,indi] = max(PCorrect(:)); [indBestC, indBestSigma] = ind2sub(size(PCorrect),indi);
CBest= CList(indBestC); sigmaBest= sigmaList(indBestSigma); 
SVMBest = fitcsvm(x',l','BoxConstraint',CBest,'KernelFunction','gaussian','KernelScale',sigmaBest);
%Labels of training data using the trained SVM
d = SVMBest.predict(x')'; 
%Find training samples that are incorrectly classified by the trained SVM
indINCORRECT = find(l.*d == -1); 
%Find training samples that are correctly classified by the trained SVM
indCORRECT = find(l.*d == 1); 
ind00 = find(d==-1 & l==-1); 
ind10 = find(d==1 & l==-1); 
ind01 = find(d==-1 & l==1);
ind11 = find(d==1 & l==1);
figure(7), 
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,
%Empirical estimate of training error probability 
pTrainingError = length(indINCORRECT)/N, 
%Grid search (unnecessary for linear kernels)
Nx = 10010; Ny = 9900; xGrid = linspace(-10,10,Nx); yGrid = linspace(-10,10,Ny);
[h,v] = meshgrid(xGrid,yGrid); dGrid = SVMBest.predict([h(:),v(:)]); zGrid = reshape(dGrid,Ny,Nx);
figure(7), contour(xGrid,yGrid,zGrid,10); xlabel('x1'), ylabel('x2'), axis equal,
legend('Correct decisions for data from Class -1','Wrong decisions for data from Class -1','Correct decisions for data from Class +1','Equilevel contours of the discriminant function' ), 
title('Training Data with Decisions'),

%Independent Test Sample
%Labels of independent data
dindep = SVMBest.predict(xv')'; 
indINCORRECTv = find(lv.*dindep == -1); 
indCORRECTv = find(lv.*dindep == 1);
pTrainingErrorv = length(indINCORRECTv)/N,
ind00 = find(dindep==-1 & lv==-1); 
ind10 = find(dindep==1 & lv==-1); 
ind01 = find(dindep==-1 & lv==1);
ind11 = find(dindep==1 & lv==1);
figure(8), 
plot(xv(1,ind00),xv(2,ind00),'og'); hold on,
plot(xv(1,ind10),xv(2,ind10),'or'); hold on,
plot(xv(1,ind01),xv(2,ind01),'+r'); hold on,
plot(xv(1,ind11),xv(2,ind11),'+g'); hold on,
axis equal,
%Grid search (unnecessary for linear kernels)
Nx = 10010; Ny = 9900; xGrid = linspace(-10,10,Nx); yGrid = linspace(-10,10,Ny);
[h,v] = meshgrid(xGrid,yGrid); dGrid = SVMBest.predict([h(:),v(:)]); zGrid = reshape(dGrid,Ny,Nx);
figure(8),contour(xGrid,yGrid,zGrid,10); xlabel('x1'), ylabel('x2'), axis equal,
legend('Correct decisions for data from Class -1','Wrong decisions for data from Class -1','Correct decisions for data from Class +1','Equilevel contours of the discriminant function' ), 
title('Validation Data with Decisions'),

%%%
function x = randGaussian(N,mu,Sigma)
%Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z =  randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);
end

%Defining evalGaussian used in script
function g = evalGaussian(x,mu,Sigma)
%Evaluates the Gaussian pdf N(mu,Sigma) at each column of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end