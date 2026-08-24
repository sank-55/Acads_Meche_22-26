%% Assignment of ACA class on 24/08/2026
A= [3,1;91,0];
C=[1 0];
B= [0;1];
p_sys = [-2-5i, -2+5i];
p_obs = [-12, -12];
K=acker(A,B,p_sys)
k_ = acker(A',C',p_obs);
Ke = k_'
% Augmented system 
A_aug = [A-B*K,B*K;zeros(2,2),A-Ke*C]
% Solving 
T=1; t=0:0.01:T; N=T/0.01;
x=zeros(4,N+1); % contains all the x1,x2,e1,e2
x(1,1)=1; x(3,1)=0.5;
% Simulate the augmented system response
for l = 1:N
    % State transition for augmented system
    Phi_aug = expm(A_aug * (t(l)));
    x(:, l+1) = Phi_aug * x(:, l);
end
% Plotting all the dynamics of x1,x2,e1,e2
figure;

% Overall title
sgtitle('Signals and Errors');

% Determine number of rows (expect at least 4)
nRows = size(x,1);
nPlot = min(nRows,4);

for k = 1:nPlot
    subplot(4,1,k);
    plot(t, x(k,:), 'LineWidth', 1.5);
    grid on;
    box on;
    switch k
        case 1
            title('x1 (Signal)');
            ylabel('x1');
        case 2
            title('x2 (Signal)');
            ylabel('x2');
        case 3
            title('x3 = Error of x1');
            ylabel('Error x1');
        case 4
            title('x4 = Error of x2');
            ylabel('Error x2');
    end
    if k == 4
        xlabel('Time');
    end
end

% Link x-axes and tighten spacing
linkaxes(findall(gcf,'Type','axes'), 'x');
set(gcf,'Units','normalized');
