
function tour2=NeighborCreation(tour1)
pSwap=0.2;
pReversion=0.5;
pInsertion=1-pSwap-pReversion;
p=[pSwap pReversion pInsertion];
METHOD=RouletteWheel(p);
switch METHOD
case 1
% Swap
tour2=Swap(tour1);       
case 2
% Reversion
tour2=Reversion(tour1);     
case 3
% Insertion
tour2=Insertion(tour1);  
end
end