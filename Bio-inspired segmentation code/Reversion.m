function tour2=Reversion(tour1)
n=numel(tour1);    
I=randsample(n,2);    
i1=min(I);
i2=max(I);    
tour2=tour1;
tour2(i1:i2)=tour1(i2:-1:i1);    
end