
function tour2=Insertion(tour1)
n=numel(tour1);   
I=randsample(n,2);    
i1=I(1);
i2=I(2);    
if i1<i2
tour2=tour1([1:i1-1 i1+1:i2 i1 i2+1:end]);
else
tour2=tour1([1:i2 i1 i2+1:i1-1 i1+1:end]);
end

end