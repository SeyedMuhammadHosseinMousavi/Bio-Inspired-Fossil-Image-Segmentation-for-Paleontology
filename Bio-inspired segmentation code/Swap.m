
function tour2=Swap(tour1)
n=numel(tour1);    
I=randsample(n,2);
i1=I(1);
i2=I(2);    
tour2=tour1;
tour2([i1 i2])=tour1([i2 i1]);    
end