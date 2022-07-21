%%************************************************************************
%% Bio-Inspired Fossil Image Segmentation for Paleontology
%% Developed by Seyed Muhammad Hossein Mousavi between Jan to July of 2022.
% This code could be used as a tool for paleontologist.
% This is demo version of:
%------------------------------------------------------------------------
%% Mousavi, S. M. H. (2018). Bio-Inspired Fossil Image Segmentation for 
%% Paleontology. International Journal of Mechatronics, Electrical and 
%% Computer Technology (IJMEC), 12(45), 5243-5249.
%------------------------------------------------------------------------
% So, please cite after use.
% Paper link:
% https://www.aeuso.org/includes/files/articles/Vol12_Iss45_5243-5249_Bio-Inspired_Fossil_Image_Segmentat.pdf
%% The code is consisted of following steps:
% 1.Evolution Strategy (ES) Histogram Equalization
% Histogram equalization is a method in image processing of
% contrast adjustment using the image's histogram
% 2.Cultural Algorithm (CA) Image Quantization
% Quantization, involved in image processing, is a lossy compression technique
% achieved by compressing a range of values to a single quantum value.
% When the number of discrete symbols in a given stream is reduced, the 
% stream becomes more compressible.
% 3.Simulated Annealing (SA) Edge Detection
% Edge detection is used to identify points 
% in a digital image with discontinuities, simply to say, sharp changes in 
% the image brightness.
% 4.Particle Swarm Optimization (PSO) Image Segmentation
% Image segmentation is the process of partitioning a digital image into
% multiple image segments, also known as image regions or image objects
% 5.Final Overlay
% F-Score performance metric evaluates the system VS ground truth
% Hope this code could help you. Let's dive in =>
%%======================================================================

%% 1.Evolution Strategy (ES) Histogram Equalization
% Clearing Things
clc;
clear;
close all;
warning ('off');
%% Loading Data
img=imread('tst2.jpg');
imcolor=img;
img=rgb2gray(img);
% Target Histogram
Data=[0,1,3,5,7,9,10,11,12,13,15,17,19,20,22,8,9,10,1,3,6,33,34,35,2];
% Creating Inputs and Targets
Delays = [1];
Data=Data';
[Inputs, Targets] = CreateTargets(Data',Delays);
data.Inputs=Inputs;
data.Targets=Targets;
% Making Data
Inputs=data.Inputs';
Targets=data.Targets';
Targets=Targets(:,1);
nSample=size(Inputs,1);
% Creating Train Vector
pTrain=1.0;
nTrain=round(pTrain*nSample);
TrainInputs=Inputs(1:nTrain,:);
TrainTargets=Targets(1:nTrain,:);
% Making Final Data Struct
data.TrainInputs=TrainInputs;
data.TrainTargets=TrainTargets;
% Making Data
Inputs=data.Inputs';
Targets=data.Targets';
Targets=Targets(:,1);
nSample=size(Inputs,1);
% Creating Train Vector
pTrain=1.0;
nTrain=round(pTrain*nSample);
TrainInputs=Inputs(1:nTrain,:);
TrainTargets=Targets(1:nTrain,:);
% Making Final Data Struct
data.TrainInputs=TrainInputs;
data.TrainTargets=TrainTargets;
%% Basic Fuzzy Model Creation 
% Number of Clusters in FCM
ClusNum=2;
% Creating FIS
fis=GenerateFuzzy(data,ClusNum);
%% Tarining Evolution Strategy Algorithm
[BestCost,ESAlgorithmFis] = ESFCN(fis,data); 
ESbestcost=BestCost;

%% Train Output Extraction
TrTar=data.TrainTargets;
TrainOutputs=evalfis(data.TrainInputs,ESAlgorithmFis);

%% Results 
% Basic Histogram Equalization
[basiceq T1] = histeq(imcolor);
% Pre-processing
% medfilt = imsharpen(medfilt2(img,[2 2],'symmetric'));
r=imcolor(:,:,1);
g=imcolor(:,:,2);
b=imcolor(:,:,3);
medf1 = imsharpen(medfilt2(r,[2 2]));
medf2 = imsharpen(medfilt2(g,[2 2]));
medf3 = imsharpen(medfilt2(b,[2 2]));
medfilt = cat(3, medf1, medf2, medf3);

% Evolution Strategy Histogram Equalization
[HisEq, T] = histeq(medfilt,TrainOutputs);
ESHistEQ = HisEq;
% Plot Results
figure;
subplot(2,3,1)
imshow(ESHistEQ);title('ES Histogram Equalization');
subplot(2,3,2)
imhist(basiceq,128);title('Basic Image Histogram ');
subplot(2,3,3)
imhist(ESHistEQ,128);title('ES Image Histogram ');
subplot(2,3,4)
plot((0:255)/255,T1);title('Basic Transformation Curve');
subplot(2,3,5)
plot((0:255)/255,T);title('ES Transformation Curve');
subplot(2,3,6)
plot(ESbestcost,'k-','LineWidth',1);
title('Evolution Strategy Algorithm Training');
xlabel('ES Iteration Number','FontSize',10,...
'FontWeight','bold','Color','k');
ylabel('ES Best Cost Result','FontSize',10,...
'FontWeight','bold','Color','k');legend({'Evolution Strategy Train'});


%% 2.Cultural Algorithm (CA) Image Quantization
%% ---------------------------------------------------------------------
%% ---------------------------------------------------------------------
%% ---------------------------------------------------------------------
%% ---------------------------------------------------------------------
%% Data Load and Preparation
img=ESHistEQ;
img=im2double(img);
% Separating color channels
R=img(:,:,1);
G=img(:,:,2);
B=img(:,:,3);
% Reshaping each channel into a vector and combine all three channels
X=[R(:) G(:) B(:)];

%% Starting CA Clustering
k = 10; % Number of Colors (cluster centers)

%---------------------------------------------------
CostFunction=@(m) ClusterCost(m, X);     % Cost Function
VarSize=[k size(X,2)];           % Decision Variables Matrix Size
nVar=prod(VarSize);              % Number of Decision Variables
VarMin= repmat(min(X),k,1);      % Lower Bound of Variables
VarMax= repmat(max(X),k,1);      % Upper Bound of Variables
% Cultural Algorithm Settings
MaxIt = 10;                       % Maximum Number of Iterations
nPop = 30;                        % Population Size
pAccept = 0.35;                   % Acceptance Ratio
nAccept = round(pAccept*nPop);    % Number of Accepted Individuals
alpha = 0.3;
beta = 0.5;
% Start
% Initialize Culture
Culture.Situational.Cost = inf;
Culture.Normative.Min = inf(VarSize);
Culture.Normative.Max = -inf(VarSize);
Culture.Normative.L = inf(VarSize);
Culture.Normative.U = inf(VarSize);
% Empty Individual Structure
empty_individual.Position = [];
empty_individual.Cost = [];
empty_individual.Out = [];
% Initialize Population Array
pop = repmat(empty_individual, nPop, 1);
% Generate Initial Solutions
for i = 1:nPop
pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
[pop(i).Cost, pop(i).Out]= CostFunction(pop(i).Position);
end
% Sort Population
[~, SortOrder] = sort([pop.Cost]);
pop = pop(SortOrder);
% Adjust Culture using Selected Population
spop = pop(1:nAccept);
Culture = AdjustCulture(Culture, spop);
% Update Best Solution Ever Found
BestSol = Culture.Situational;
% Array to Hold Best Costs
BestCost = zeros(MaxIt, 1);
%% Cultural Algorithm Body
for it = 1:MaxIt
% Influnce of Culture
for i = 1:nPop
% % 3rd Method (using Normative and Situational components)
for j = 1:nVar
sigma = alpha*Culture.Normative.Size(j);
dx = sigma*randn;
if pop(i).Position(j)<Culture.Situational.Position(j)
dx = abs(dx);
elseif pop(i).Position(j)>Culture.Situational.Position(j)
dx = -abs(dx);
end
pop(i).Position(j) = pop(i).Position(j)+dx;
end          
[pop(i).Cost, pop(i).Out] = CostFunction(pop(i).Position);
end
% Sort Population
[~, SortOrder] = sort([pop.Cost]);
pop = pop(SortOrder);
% Adjust Culture using Selected Population
spop = pop(1:nAccept);
Culture = AdjustCulture(Culture, spop);
% Update Best Solution Ever Found
BestSol = Culture.Situational;
% Store Best Cost Ever Found
BestCost(it) = BestSol.Cost;
% Show Iteration Information
disp(['In Iteration ' num2str(it) ': CA Best Cost Is = ' num2str(BestCost(it))]);
CACenters=Res(X, BestSol);
end
disp('Cultural Algorithm Came To End');
CAlbl=BestSol.Out.ind;
CAcost=BestCost;   

%% Converting cluster centers and its indexes into image 
Z=CACenters(CAlbl',:);
R2=reshape(Z(:,1),size(R));
G2=reshape(Z(:,2),size(G));
B2=reshape(Z(:,3),size(B));
% Attaching color channels 
quantized=zeros(size(img));
quantized(:,:,1)=R2;
quantized(:,:,2)=G2;
quantized(:,:,3)=B2;
% Plot Results
figure;
subplot(2,2,1)
imshow(img); title('Original ES-HE');
subplot(2,2,2)
subimage(quantized);title('Cultural Algorithm Quantization');
subplot(2,2,3)
plot(CAcost,'-k','linewidth',1);
title('CA Train');
xlabel('CA Iteration Number');
ylabel('CA Best Cost Value');

%-------------------------------------------------------
%-------------------------------------------------------
%-------------------------------------------------------
%-------------------------------------------------------


%% 3.Simulated Annealing (SA) Edge Detection
pic=rgb2gray(quantized);
sacolor=quantized;
%-------------------------------------------------------
% Filters
polished1=[-2.2 -0.8 -0.6 ;0 0 0 ;2.2 0.8 0.6 ];
polished11=[2.2 0.8 0.6 ;0 0 0 ;-2.2 -0.8 -0.6 ];
polished111=[-0.1 -0.8 -0.6 ;0 0 0;0.1 0.8 0.6 ];
polished1111=[0.1 0.8 0.6 ;0 0 0 ;-0.1 -0.8 -0.6 ];
polished2=polished1';
polished22=polished11';
polished222=polished111';
polished2222=polished1111';
%--------------------------------------------------------
% Combining Filteres
Pol1=[polished1 polished11 polished111 polished1111];
Pol1(:,end+1)=1;
Pol2=[polished2 polished22 polished222 polished2222];
Pol2(:,end+1)=2;
PolFil=[Pol1; Pol2];
% Swap Filter Matrix Row Randomly Each Run for Productivity
PolFil_Swap = PolFil(randperm(size(PolFil, 1)), :);
%% Data Preparation
fordet=PolFil_Swap;
sizdet=size(fordet);
x=PolFil_Swap(:,1:sizdet(1,2)-1)';
t=PolFil_Swap(:,sizdet(1,2))';
nx=sizdet(1,2)-1;
nt=1;
nSample=sizdet(1,1);
% Converting Table to Struct
data.x=x;
data.t=t;
data.nx=nx;
data.nt=nt;
data.nSample=nSample;
nf=6;
% Cost Function
CostFunction=@(q) FSC(q,nf,data);
%% Simulated Annealing Parameters
MaxIt=10;      % Max Number of Iterations
MaxSubIt=3;    % Max Number of Sub-iterations
T0=5;          % Initial Temp
alpha=0.99;    % Temp Reduction Rate
% Create and Evaluate Initial Solution
sol.Position=CRS(data);
[sol.Cost, sol.Out]=CostFunction(sol.Position);
% Initialize Best Solution Ever Found
BestSol=sol;
% Array to Hold Best Cost Values
BestCost=zeros(MaxIt,1);
% Intialize Temp.
T=T0;
%% Simulated Annealing Run
for it=1:MaxIt
for subit=1:MaxSubIt
% Create and Evaluate New Solution
newsol.Position=NeighborCreation(sol.Position);
[newsol.Cost, newsol.Out]=CostFunction(newsol.Position);
% If NEWSOL is better than SOL
if newsol.Cost<=sol.Cost 
sol=newsol;
else % If NEWSOL is NOT better than SOL
DELTA=(newsol.Cost-sol.Cost)/sol.Cost;
P=exp(-DELTA/T);
if rand<=P
sol=newsol;
end
end
% Update Best Solution Ever Found
if sol.Cost<=BestSol.Cost
BestSol=sol;
end
end
% Store Best Cost Ever Found
BestCost(it)=BestSol.Cost;
SAcost=BestCost;
% Display Iteration
disp(['In Iteration Number ' num2str(it) ': SA Best Cost = ' num2str(BestCost(it))]);
% Update Temp
T=alpha*T;
end
disp('Simulated Annealing Algorithm Came To End');
%% Data Post Processing
% Extracting Data
RealData=PolFil_Swap;
% Extracting Labels
RealLbl=RealData(:,end);
FinalFeaturesInd=BestSol.Out.S;
% Sort Features
FFI=sort(FinalFeaturesInd);
% Select Final Features
SA_Features=RealData(:,FFI);
% Adding Labels
SA_Features(:,end+1)=RealLbl;
%% Applay SA Filters on Image
FinalFilt=SA_Features(:,1:end-1);
p1=imfilter(pic,FinalFilt(1:3,1:3));
p2=imfilter(pic,FinalFilt(4:6,1:3));
p3=imfilter(pic,FinalFilt(1:3,4:6));
p4=imfilter(pic,FinalFilt(4:6,4:6));
SA_Edge=rangefilt(abs(p1)+abs(p2)+abs(p3)+abs(p4));

% Plot Res
figure;
subplot(2,2,1)
imshow(sacolor);title('CA Quantized');
subplot(2,2,2)
imshow(pic);title('CA Quantized - Gray');
subplot(2,2,3)
imshow(SA_Edge,[]);title('SA Edges');
subplot(2,2,4)
plot(SAcost,'-k');
title('Simulated Annealing')
xlabel('SA Iteration Number','FontSize',12,...
'FontWeight','bold','Color','k');
ylabel('SA Best Cost Result','FontSize',12,...
'FontWeight','bold','Color','k');
legend({'SA Train'});


%% 4.PSOSA Image Segmentation
%%-------------------------------------------------------------
%%-------------------------------------------------------------
%%-------------------------------------------------------------
%%-------------------------------------------------------------
%% Reading Image
%Original
MainOrg=quantized;

r2=MainOrg(:,:,1);
g2=MainOrg(:,:,2);
b2=MainOrg(:,:,3);
medf3 = imsharpen(medfilt2(r2,[2 2]));
medf4 = imsharpen(medfilt2(g2,[2 2]));
medf5 = imsharpen(medfilt2(b2,[2 2]));
% medfiltpso = cat(3, medf3, medf4, medf5);

% MainOrg=rgb2gray(MainOrg);
% MainOrg = medfilt2(MainOrg,[2 2]);
MainOrg = cat(3, medf3, medf4, medf5);

%%-------------------------------------------------------------
Gray=rgb2gray(MainOrg);
InpMat= double(MainOrg);

%% Basics
[s1,s2,s3]=size(InpMat);
R = InpMat(:,:,1);
G = InpMat(:,:,2);
Bb = InpMat(:,:,3);
X1 = (R-min(R(:)))/(max(R(:))-min(R(:)));
X2 = (G-min(G(:)))/(max(G(:))-min(G(:)));
X3 = (Bb-min(Bb(:)))/(max(Bb(:))-min(Bb(:)));
X = [X1(:) X2(:) X3(:)];
%% Cluster Numbers
clusteres = 4;

%% Cost Function and Parameters
% Cost Function
CostFunction=@(m) CLuCosPSOSA(m, X, clusteres);  
% Decision Variables
VarSize=[clusteres size(X,2)];  
% Number of Decision Variables
nVar=prod(VarSize);
% Lower Bound of Variables
VarMin= repmat(min(X),1,clusteres);      
% Upper Bound of Variables
VarMax= repmat(max(X),1,clusteres);     
%% PSO-SA Clustering Option and Run
% PSO-SA Options
% Iterations (more value means: slower runtime but, better result)
Itr=20;

% SA solver + PSO body
SA_opts = optimoptions('simulannealbnd','display','iter','MaxTime',Itr,'PlotFcn',@pswplotbestf);
options.SwarmSize = 70;

% PSO-SA Run
disp(['SA-PSO Segmentation Is Started ... ']);
[centers, Error] = particleswarm(CostFunction, nVar,VarMin,VarMax,SA_opts);
disp(['SA-PSO Segmentation Is Ended. ']);
%% Calculate Distance Matrix
% Create the Cluster Center 
g=reshape(centers,3,clusteres)'; 
% Create a Distance Matrix
d = pdist2(X, g); 
% Assign Clusters and Find Closest Distances
[dmin, ind] = min(d, [], 2);
% Sum of Cluster Distance
WCD = sum(dmin);
% Fitness Function of Centers Sum
z=WCD; 
% Final Segmented Image
SA_Segmented=reshape(ind,s1,s2);
PSOSAuint=uint8(SA_Segmented);
ColorSeg = labeloverlay(Gray,PSOSAuint);
medgray = medfilt2(SA_Segmented,[5 5]);
redChannel = ColorSeg(:,:,1); % Red channel
greenChannel = ColorSeg(:,:,2); % Green channel
blueChannel = ColorSeg(:,:,3); % Blue channel
medcolor1 = medfilt2(redChannel,[4 6]);
medcolor2 = medfilt2(greenChannel,[4 6]);
medcolor3 = medfilt2(blueChannel,[4 6]);
medrgb = cat(3, medcolor1, medcolor2, medcolor3);
%% Plot PSO-SA Segmented Result
disp(['Error Is: ' num2str(Error)]);
%% 5.Composite and overlay all
C = imfuse(SA_Edge,ColorSeg);
Cc = imfuse(C,B);
%% Final plots
figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,3,1)
imshow(img,[]);title('Original','FontSize', 15,'color','r');
subplot(2,3,2)
imshow(ESHistEQ,[]);title('ES Histogram Equalization','FontSize', 15,'color','r');
subplot(2,3,3)
imshow(quantized);title('Cultural Algorithm Quantization','FontSize', 15,'color','r');
subplot(2,3,4)
imshow(SA_Edge,[]);title('Simulated Annealing Edges','FontSize', 15,'color','r');
subplot(2,3,5)
imshow(ColorSeg,[]);
title(['PSO-SA Color Segmented, Clusters = ' num2str(clusteres)],'FontSize', 15,'color','r');
subplot(2,3,6)
imshow(Cc,[]);
title('Final Overlay','FontSize', 15,'color','r');

%% Statistics
% Ground Truth
bwimg=im2bw(quantized);%imshow(bwimg);
% Proposed
segbw=im2bw(ColorSeg);%imshow(segbw);
% fscore
F_score = bfscore(segbw, bwimg)


