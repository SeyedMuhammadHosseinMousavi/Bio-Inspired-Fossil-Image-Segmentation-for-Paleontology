function bestfis=CulturalFCN(fis,data)
% Variables
p0=GettingFuzzyParameters(fis);
Problem.CostFunction=@(x) FuzzyCost(x,fis,data);
Problem.nVar=numel(p0);
alpha=1;
VarMin = -10;         % Decision Variables Lower Bound
VarMax = 10;          % Decision Variables Upper Bound
Problem.VarMin=-(10^alpha);
Problem.VarMax=10^alpha;
%
% Cultural Algorithm Parameters
pAccept = 0.40;
nPop = 8;
Params.MaxIt = 50;       % Maximum Number of Iterations
Params.pAccept = 0.40;  
Params.nPop = nPop; 
Params.nAccept = round(pAccept*nPop);    % Number of Accepted Individuals
Params.alpha = 0.3;
Params.beta = 0.5;
%
% Starting Cultural Algorithm
results=RunCulturalFCN(Problem,Params);
% Getting the Results
p=results.BestSol.Position.*p0;
bestfis=FuzzyParameters(fis,p);
end
%%----------------------------------------------
function results=RunCulturalFCN(Problem,Params)
disp('Starting Cultural Algorithm Training');
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
%% Cultural Algorithm Parameters
MaxIt=Params.MaxIt;       % Maximum Number of Iterations
pAccept=Params.pAccept;                  % Acceptance Ratio
nPop=Params.nPop; 
nAccept=Params.nAccept;    % Number of Accepted Individuals
alpha=Params.alpha;
beta=Params.beta;
%------------------------------------------------------
%% Second Stage
% Initialize Culture
Culture.Situational.Cost = inf;
Culture.Normative.Min = inf(VarSize);
Culture.Normative.Max = -inf(VarSize);
Culture.Normative.L = inf(VarSize);
Culture.Normative.U = inf(VarSize);
% Empty Individual Structure
empty_individual.Position = [];
empty_individual.Cost = [];
% Initialize Population Array
pop = repmat(empty_individual, nPop, 1);
% Generate Initial Solutions
for i = 1:nPop
pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
pop(i).Cost = CostFunction(pop(i).Position);
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
%
%% Cultural Algorithm Main Body
for it = 1:MaxIt
    % Influnce of Culture
    for i = 1:nPop
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
        pop(i).Cost = CostFunction(pop(i).Position);
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
    disp(['In Iteration ' num2str(it) ' Out Of ' num2str(MaxIt) ' : CA Fittest Value IS = ' num2str(BestCost(it))]);
end
%------------------------------------------------
disp('Cultural Algorithm Came To End');
% Store Result
results.BestSol=BestSol;
results.BestCost=BestCost;
% Plot Cultural Algorithm Training Stages
figure;
set(gcf, 'Position',  [600, 300, 500, 200])
plot(BestCost,'-',...
'LineWidth',2,...
'MarkerSize',3,...
'MarkerEdgeColor','g',...
'Color',[0.1,0.1,0.1]);
title('Cultural Algorithm Algorithm Training')
xlabel('CA Iteration Number','FontSize',10,...
'FontWeight','bold','Color','k');
ylabel('CA Best Cost Result','FontSize',10,...
'FontWeight','bold','Color','k');
legend({'Cultural Algorithm Train'});
end




