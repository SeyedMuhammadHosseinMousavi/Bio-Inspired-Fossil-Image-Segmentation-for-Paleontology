function [BestCost,bestfis]=ESFCN(fis,data)
% Variables
p0=GettingFuzzyParameters(fis);
Problem.CostFunction=@(x) FuzzyCost(x,fis,data);
Problem.nVar=numel(p0);
alpha=1;
nVar = 10;            % Number of Unknown (Decision) Variables
VarMin = -10;         % Decision Variables Lower Bound
VarMax = 10;          % Decision Variables Upper Bound
Problem.VarMin=-(10^alpha);
Problem.VarMax=10^alpha;
%
% Evolution Strategy Parameters
% Maximum Number of Iterations
Params.MaxIt = 10;
% Population Size (and Number of Offsprings)
Params.lambda = (4+round(3*log(nVar)))*10;
% Number of Parents
Params.mu = round(Params.lambda/2);
% Parent Weights
Params.w = log(Params.mu+0.5)-log(1:Params.mu);
Params.w = Params.w/sum(Params.w);
% Number of Effective Solutions
Params.mu_eff = 1/sum(Params.w.^2);
% Step Size Control Parameters (c_sigma and d_sigma);
Params.sigma0 = 0.3*(VarMax-VarMin);
Params.cs = (Params.mu_eff+2)/(nVar+Params.mu_eff+5);
Params.ds = 1+Params.cs+2*max(sqrt((Params.mu_eff-1)/(nVar+1))-1, 0);
Params.ENN = sqrt(nVar)*(1-1/(4*nVar)+1/(21*nVar^2));
% Covariance Update Parameters
Params.cc = (4+Params.mu_eff/nVar)/(4+nVar+2*Params.mu_eff/nVar);
Params.c1 = 2/((nVar+1.3)^2+Params.mu_eff);
Params.alpha_mu = 2;
Params.cmu = min(1-Params.c1, Params.alpha_mu*(Params.mu_eff-2+1/Params.mu_eff)/((nVar+2)^2+Params.alpha_mu*Params.mu_eff/2));
Params.hth = (1.4+2/(nVar+1))*Params.ENN;
%
% Starting Evolution Strategy
[BestCost,results]=RunESFCN(Problem,Params);
% Getting the Results
p=results.BestSol.Position.*p0;
bestfis=FuzzyParameters(fis,p);
end
%%----------------------------------------------
function [BestCost,results]=RunESFCN(Problem,Params)
disp('Starting Evolution Strategy Training');
%------------------------------------------------
% Cost Function
CostFunction=Problem.CostFunction;
% Number of Decision Variables
nVar=Problem.nVar;
% Size of Decision Variables Matrixv
VarSize=[1 nVar];
% Lower Bound of Variables
VarMin=Problem.VarMin;
% Upper Bound of Variables
VarMax=Problem.VarMax;
% Some Change
if isscalar(VarMin) && isscalar(VarMax)
dmax = (VarMax-VarMin)*sqrt(nVar);
else
dmax = norm(VarMax-VarMin);
end
%--------------------------------------------
%% Evolution Strategy Parameters
MaxIt = Params.MaxIt;
lambda =Params.lambda;
mu = round(Params.lambda/2);
w = log(Params.mu+0.5)-log(1:Params.mu);
w = Params.w/sum(Params.w);
mu_eff = 1/sum(Params.w.^2);
sigma0 = 0.3*(VarMax-VarMin);
cs = (Params.mu_eff+2)/(nVar+Params.mu_eff+5);
ds = 1+Params.cs+2*max(sqrt((Params.mu_eff-1)/(nVar+1))-1, 0);
ENN = sqrt(nVar)*(1-1/(4*nVar)+1/(21*nVar^2));
cc = (4+Params.mu_eff/nVar)/(4+nVar+2*Params.mu_eff/nVar);
c1 = 2/((nVar+1.3)^2+Params.mu_eff);
alpha_mu =Params.alpha_mu;
cmu = min(1-Params.c1, Params.alpha_mu*(Params.mu_eff-2+1/Params.mu_eff)/((nVar+2)^2+Params.alpha_mu*Params.mu_eff/2));
hth = (1.4+2/(nVar+1))*Params.ENN;
%------------------------------------------------------
%% Second Stage
ps = cell(MaxIt, 1);
pc = cell(MaxIt, 1);
C = cell(MaxIt, 1);
sigma = cell(MaxIt, 1);
ps{1} = zeros(VarSize);
pc{1} = zeros(VarSize);
C{1} = eye(nVar);
sigma{1} = sigma0;
empty_individual.Position = [];
empty_individual.Step = [];
empty_individual.Cost = [];
M = repmat(empty_individual, MaxIt, 1);
M(1).Position = unifrnd(VarMin, VarMax, VarSize);
M(1).Step = zeros(VarSize);
M(1).Cost = CostFunction(M(1).Position);
BestSol = M(1);
BestCost = zeros(MaxIt, 1);
%
%% Evolution Strategy Main Body
for g = 1:MaxIt
% Generate Samples
pop = repmat(empty_individual, lambda, 1);
for i = 1:lambda
% Generating Sample
pop(i).Step = mvnrnd(zeros(VarSize), C{g});
pop(i).Position = M(g).Position + sigma{g}*pop(i).Step;
% Applying Bounds
pop(i).Position = max(pop(i).Position, VarMin);
pop(i).Position = min(pop(i).Position, VarMax);
% Evaluation
pop(i).Cost = CostFunction(pop(i).Position);
% Update Best Solution Ever Found
if pop(i).Cost<BestSol.Cost
BestSol = pop(i);
end
end
% Sort Population
Costs = [pop.Cost];
[Costs, SortOrder] = sort(Costs);
pop = pop(SortOrder);
% Save Results
BestCost(g) = BestSol.Cost;
% Display Results
disp(['In Iteration ' num2str(g) ' Out Of ' num2str(MaxIt) ' : ES Fittest Value IS = ' num2str(BestCost(g))]);
if g == MaxIt
break;
end
% Update Mean
M(g+1).Step = 0;
for j = 1:mu
M(g+1).Step = M(g+1).Step+w(j)*pop(j).Step;
end
M(g+1).Position = M(g).Position + sigma{g}*M(g+1).Step;
% Applying Bounds
M(g+1).Position = max(M(g+1).Position, VarMin);
M(g+1).Position = min(M(g+1).Position, VarMax);
% Evaluation
M(g+1).Cost = CostFunction(M(g+1).Position);
% Update Best Solution Ever Found
if M(g+1).Cost < BestSol.Cost
BestSol = M(g+1);
end
% Update Step Size
ps{g+1} = (1-cs)*ps{g}+sqrt(cs*(2-cs)*mu_eff)*M(g+1).Step/chol(C{g})';
sigma{g+1} = sigma{g}*exp(cs/ds*(norm(ps{g+1})/ENN-1))^0.3;
% Update Covariance Matrix
if norm(ps{g+1})/sqrt(1-(1-cs)^(2*(g+1)))<hth
hs = 1;
else
hs = 0;
end
delta = (1-hs)*cc*(2-cc);
pc{g+1} = (1-cc)*pc{g}+hs*sqrt(cc*(2-cc)*mu_eff)*M(g+1).Step;
C{g+1} = (1-c1-cmu)*C{g}+c1*(pc{g+1}'*pc{g+1}+delta*C{g});
for j = 1:mu
C{g+1} = C{g+1}+cmu*w(j)*pop(j).Step'*pop(j).Step;
end
% If Covariance Matrix is not Positive Defenite or Near Singular
[V, E] = eig(C{g+1});
if any(diag(E)<0)
E = max(E, 0);
C{g+1} = V*E/V;
end
end
disp('Evolution Strategy Algorithm Came To End');
% Store Result
results.BestSol=BestSol;
results.BestCost=BestCost;
end






