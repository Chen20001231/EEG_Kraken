function [feature] = feature_extraction(data)

[cd1, cd2, cd3, cd4, cd5, cd6, cd7, ca1] = wavelet(data);
 
feature = zeros(2*nargin ,1);

feature(1) = energy_ex(cd1);
feature(2) = entropy_ex(cd1);


feature(3) = energy_ex(cd2);
feature(4) = entropy_ex(cd2);

feature(5) = energy_ex(cd3);
feature(6) = entropy_ex(cd3);

feature(7) = energy_ex(cd4);
feature(8) = entropy_ex(cd4);

feature(9) = energy_ex(cd5);
feature(10) = entropy_ex(cd5);

feature(11) = energy_ex(cd6);
feature(12) = entropy_ex(cd6);

feature(13) = energy_ex(ca1);
feature(14) = entropy_ex(ca1);

feature(15) = std_ex(cd1);
feature(16) = std_ex(cd2);
feature(17) = std_ex(cd3);
feature(18) = std_ex(cd4);
feature(19) = std_ex(cd5);
feature(20) = std_ex(cd6);
feature(21) = std_ex(ca1);

feature(22) = kurtosis_ex(cd1);
feature(23) = kurtosis_ex(cd2);
feature(24) = kurtosis_ex(cd3);
feature(25) = kurtosis_ex(cd4);
feature(26) = kurtosis_ex(cd5);
feature(27) = kurtosis_ex(cd6);
feature(28) = kurtosis_ex(ca1);

feature(29) = skewness_ex(cd1);
feature(30) = skewness_ex(cd2);
feature(31) = skewness_ex(cd3);
feature(32) = skewness_ex(cd4);
feature(33) = skewness_ex(cd5);
feature(34) = skewness_ex(cd6);
feature(35) = skewness_ex(ca1);

feature(36) = energy_ex(cd7);
feature(37) = entropy_ex(cd7);


end

