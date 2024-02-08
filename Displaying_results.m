
% D = 'results\industrial-robot\repeat-9'; % A is a struct ... first elements are '.' and '..' used for navigation.
% for k = 3:length(D) % avoid using the first ones
%     currD = D(k).name; % Get the current subdirectory name
%       % Run your function. Note, I am not sure on how your function is written,
%       % but you may need some of the following
%       cd(currD) % change the directory (then cd('..') to get back)
% 
% 
%       fList = dir(currD); % Get the file list in the subdirectory
%   end
%s= strcat()
%h5disp('results\industrial-robot\repeat-9\ReLiNet-128-4\scores-test-w_60-h_60.hdf5', '/horizon-1/nrmse/score/')
%figure; hold on

% for k = 1:6
%     for i = [1 15 30 45 60]
%         dataset = strcat('/horizon-', int2str(i), '/nrmse/score/');
%         data = h5read('results\industrial-robot\repeat-9\ReLiNet-128-4\scores-test-w_60-h_60.hdf5', dataset);
%         plot(i, data, '.'); hold on
%     end
% end
set(groot,'DefaultFigureGraphicsSmoothing','on')
for i = 1:60
    dataset = strcat('/horizon-', int2str(i), '/nrmse/score/');
    data(:, i) = h5read('results\industrial-robot\repeat-9\QLag-15\scores-test-w_60-h_60.hdf5', dataset);
end
for k = 1:6
    plot(data(k, :), 'DisplayName',strcat('y', int2str(k))); hold on
end
hold off
legend show