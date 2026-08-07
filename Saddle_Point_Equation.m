clear;
clc;
close all;
warning('off', 'MATLAB:integral:NonFiniteValue');
warning('off', 'MATLAB:fzero:NAx');

format long;

File_Name = sprintf(mfilename);
Str = [ File_Name, '_' ];

EPS = 1e-12;
Para.EPS = EPS;

Spectrum = 'Rad';
% Spectrum = 'Quad';
% Spectrum = 'Jacobi';
% Spectrum = 'Gauss';
Para.Spectrum = Spectrum;

Gamma = 2;
Para.Gamma = Gamma;

C = 2;
Para.C = C;

Kappa = 1;
Para.Kappa = Kappa;

% 优先分布类型:
% Opt_U = 'Rad';
Opt_U = '2Points';
Para.Opt_U = Opt_U;

epsilon = 1 / 12;
Para.epsilon = epsilon;

rho = 0.4;
Para.rho = rho;

% Opt_V = 'Rad';
Opt_V = 'Gauss_Ber';
Para.Opt_V = Opt_V;

Alpha = 2.0;
Para.Alpha = Alpha;

Delta = 1 / Alpha; % M / N
Para.Delta = Delta;

% 获取势函数相关参数
[X_Min, X_Max, Rho, H, S, S0, VP, Str] = Choose_Potential(Para, Str);

Para.X_Min = X_Min;
Para.X_Max = X_Max;
Para.Rho = Rho;
Para.H = H;
Para.S = S;
Para.S0 = S0;
Para.VP = VP;

% 获取先验分布的Overlap函数
[D_U, E_Log_P_U, Str] = Choose_Prior(Para, Opt_U, Str);
[D_V, E_Log_P_V, Str] = Choose_Prior(Para, Opt_V, Str);

Para.D_U = D_U;
Para.E_Log_P_U = E_Log_P_U;
Para.D_V = D_V;
Para.E_Log_P_V = E_Log_P_V;

Str = [ Str, 'A-', num2str(Alpha), '_' ];

Theta_Interval = 0.001;
% Theta_Interval = 0.01;

Theta_Array = [ 0.28 : Theta_Interval : 0.34 ];
Str = [ Str, 'T-', num2str(Theta_Array(1)), '-', num2str(Theta_Interval), '-', num2str(Theta_Array(end)) ];

Num_Theta = length(Theta_Array);

% Interval = 100;
Interval = 1000;
Para.Interval = Interval;

Str = [ Str, 'I-', num2str(Interval) ];

Q_Num = 120;
Q_Contour = 256;

Para.Q_Num = Q_Num;
Para.Q_Contour = Q_Contour;

tmp = 1e-5;

W_U_Array = linspace(0 + tmp, 1 - tmp, Interval);
W_V_Array = linspace(0 + tmp, 1 - tmp, Interval);

Residual_Array = zeros(Num_Theta, Interval, Interval);
Solution_Array = cell(Num_Theta, 1);

Num_Bar = 40; % A slightly shorter bar looks better in plain list

tic; % Start the timer outside the loop

for t = 1 : Num_Theta

	Theta = Theta_Array(t);
	Lambda = Theta / sqrt(Alpha);

	for i = 1 : length(W_U_Array)
		for j = 1 : length(W_V_Array)

			W_U = W_U_Array(i);
			W_V = W_V_Array(j);

			Input = [W_U; W_V];
			Output = Compute_Residual(Para, Theta, Input);

			Res_1 = Output(1);
			Res_2 = Output(2);

			Res = Res_1^(2) + Res_2^(2);

			Residual_Array(t, i, j) = Res;

		end
	end

	Residual = squeeze( Residual_Array(t, :, :) );

	Index_Array = Find_Position(Residual);

	Num_Index = size(Index_Array, 2);

	Solution = zeros(Num_Index, 14);

	for i = 1 : Num_Index

		Row = Index_Array(1, i);
		Col = Index_Array(2, i);

		fprintf(' ------------------------------------- \n');

		fprintf( ...
			'Find_Position | t: %d, Theta: %d, i : %d, Row: %d, Col: %d\n', ...
			t, Theta, i, Row, Col ...
		);

		Solution(i, 1) = Row;
		Solution(i, 2) = Col;

		W_U = W_U_Array(Row);
		W_V = W_V_Array(Col);

		Solution(i, 3) = W_U;
		Solution(i, 4) = W_V;

		fprintf('W_U: %d\n', W_U);
		fprintf('W_V: %d\n', W_V);

		Input = [W_U; W_V];

		% 优化配置：关掉迭代中的打印（因为你的函数内部已经有fprintf了）
		Opt = optimoptions('fsolve', 'Display', 'off', 'TolFun', 1e-14);

		% 调用fsolve
		[Sol, ~, ~] = fsolve( @ (x) Compute_Residual(Para, Theta, x), Input, Opt);

		Final_W_U = Sol(1);
		Final_W_V = Sol(2);

		Solution(i, 5) = Final_W_U;
		Solution(i, 6) = Final_W_V;

		Final_M_U = D_U(Final_W_U);
		Final_M_V = D_V(Final_W_V);

		Solution(i, 7) = Final_M_U;
		Solution(i, 8) = Final_M_V;

		fprintf( ...
			'Final_W_U: %d, Final_W_V: %d, Final_M_U: %d, Final_M_V: %d\n', ...
			Final_W_U, Final_W_V, Final_M_U, Final_M_V ...
		);

		Final_FE = Compute_Free_Energy(Para, Lambda, Final_W_U, Final_W_V, Final_M_U, Final_M_V);

		Solution(i, 9) = Final_FE;

		fprintf('Final_FE: %d\n', Final_FE);

		Input = [Final_W_U; Final_W_V];
		Output = Compute_Residual(Para, Theta, Input);

		Res_1 = Output(1);
		Res_2 = Output(2);

		Solution(i, 10) = Res_1;
		Solution(i, 11) = Res_2;

		Norm_Res_1 = norm(Res_1) / norm(Final_W_U);
		Norm_Res_2 = norm(Res_2) / norm(Final_W_V);

		Solution(i, 12) = Norm_Res_1;
		Solution(i, 13) = Norm_Res_2;

		Res = Res_1^(2) + Res_2^(2);

		Solution(i, 14) = Res;

		fprintf( ...
			'Res_1: %d, Res_2: %d, Res: %d\n', ...
			Res_1, Res_2, Res ...
		);

		fprintf( ...
			'Norm_Res_1: %d, Norm_Res_2: %d\n', ...
			Norm_Res_1, Norm_Res_2 ...
		);

	end

	Solution_Array{t} = Solution;

	Progress = t / Num_Theta;
	Percent = Progress * 100;
	elapsedTime = toc;  % Capture current time
	
	% Create the static bar [#####.....]
	Num_Hash = round(Progress * Num_Bar);
	Num_Dot = Num_Bar - Num_Hash;
	Bar_Str = [ '[', repmat( '#', 1, Num_Hash ), repmat( '.', 1, Num_Dot ), ']' ];

	% Direct print (No backspaces)
	% \n ensures every update starts on a NEW line
	fprintf( ...
		'%s %3.0f%% | Theta: %d | Lambda: %d | Position: %d | Elapsed: %d seconds\n', ...
		Bar_Str, Percent, Theta, Lambda, Num_Index, elapsedTime ...
	);

end

save([Str, '.mat'])

% 核心功能函数定义
function [X_Min, X_Max, Rho, H, S, S0, VP, Str] = Choose_Potential(Para, Str)

	EPS = Para.EPS;

	Spectrum = Para.Spectrum;

	Alpha = Para.Alpha;

	Gamma = Para.Gamma;

	C = Para.C;

	Kappa = Para.Kappa;

	if strcmp(Spectrum, 'Rad')
		X_Min = 1;
		X_Max = 3;

		Mid = (X_Min + X_Max) / 2;

		Rho = @ (x) 2 / pi * sqrt( max( 0, (x - X_Min) .* (X_Max - x) ) );
		H = @ (x) 2 / pi * (x - Mid);
		S = @ (x) 2 * ( x - Mid + sqrt( (x - X_Min) .* (x - X_Max) ) );
		S0 = S(0);
		VP = @ (x) 2 * pi * H(x) + (Alpha - 1) ./ x;
		Str = [ Str, 'Rad-', num2str(X_Min), '-', num2str(X_Max), '_' ];

	elseif strcmp(Spectrum, 'Quad')

		F_Term = (Alpha - 1)^(2) / (Alpha + 1)^(2);

		S_Term = sqrt( 1 + 3 * F_Term );

		d_0 = sqrt( (Alpha + 1) / 3 * (1 + S_Term) );

		Delta_0 = sqrt( 2 * (Alpha + 1) / 3 * (2 - S_Term) );

		X_Min = (d_0 - Delta_0) / sqrt(Gamma);
		X_Max = (d_0 + Delta_0) / sqrt(Gamma);

		Mid = (X_Min + X_Max) / 2;

		% 定义密度函数Rho(x) (公式 43)
		% 这里的Rho必须保证在[X_Min, X_Max]之外为0
		Rho = @ (x) Gamma / (2 * pi) * sqrt( ...
			max( 0, (x - X_Min) .* (X_Max - x) ) ...
		) .* ( ...
			1 + Mid ./ x ...
		);

		H = @ (x) 1 / (2 * pi) * ( ...
			Gamma * x - (Alpha - 1) ./ x ...
		);

		S = @ (x) 1 / 2 * ( ...
			Gamma * x - (Alpha - 1) ./ x ...
		) ...
		+ Gamma / 2 * sqrt( ...
			(x - X_Min) .* (x - X_Max) ...
		) .* ( ...
			1 + Mid ./ x ...
		);

		S0 = - ( ...
			Delta_0^(2) * sqrt(Gamma) ...
		) / ( ...
			2 * sqrt( d_0^(2) - Delta_0^(2) ) ...
		);

		VP = @ (x) Gamma * x;

		Str = [ Str, 'Quad-', num2str(Gamma), '-', num2str(X_Min), '-', num2str(X_Max), '_' ];

	elseif strcmp(Spectrum, 'Jacobi')

		%计算d_0, Delta_0, X_Min, X_Max (公式 51, 52)
		d_0 = 1 / 2 * ( ...
			1 + (Alpha - 1)^(2) / (Alpha + C)^(2) - (C - 1)^(2) / (Alpha + C)^(2) ...
		);

		Delta_0 = sqrt( ...
			d_0^(2) - (Alpha - 1)^(2) / (Alpha + C)^(2) ...
		);

		X_Min = Kappa * ( d_0 - Delta_0 );
		X_Max = Kappa * ( d_0 + Delta_0 );

		Rho = @(x) (Alpha + C) / (2 * pi) * sqrt( ...
			max( 0, (x - X_Min) .* (X_Max - x) ) ...
		) ./ ( x .* (Kappa - x) );

		H = @ (x) 1 / (2 * pi) * ( ...
			(C - 1) ./ (Kappa - x) - (Alpha - 1) ./ x ...
		);

		S = @ (x) 1 / 2 * ( ...
			(C - 1) ./ (Kappa - x) - (Alpha - 1) ./ x ...
		) ...
		+ (Alpha + C) / 2 * sqrt( ...
			(x - X_Min) .* (x - X_Max) ...
		) ./ ( x .* (Kappa - x) );

		S0 = (C + Alpha - 2) / (2 * Kappa) ...
		- (Alpha - 1) * (X_Min + X_Max) / (4 * X_Min * X_Max);

		VP = @ (x) (C - 1) ./ (Kappa - x);

		Str = [ Str, 'Jacobi-', num2str(C), '-', num2str(Kappa), '-', num2str(X_Min), '-', num2str(X_Max), '_' ];

	elseif strcmp(Spectrum, 'Gauss')

		X_Min = ( 1 - sqrt(Alpha) )^(2);
		X_Max = ( 1 + sqrt(Alpha) )^(2);

		Rho = @(x) 1 / (2 * pi) * sqrt( ...
			max( 0, (x - X_Min) .* (X_Max - x) ) ...
		) ./ x;

		H = @ (x) 1 / (2 * pi) * ( ...
			1 - (Alpha - 1) ./ x ...
		);

		S = @ (x) 1 / 2 * ( ...
			1 - (Alpha - 1) ./ x ...
		) ...
		+ 1 / 2 * sqrt( ...
			(x - X_Min) .* (x - X_Max) ...
		) ./ x;

		S0 = 1 / 2 - (X_Min + X_Max) / (4 * (Alpha - 1));

		VP = @ (x) 1;

		Str = [ Str, 'Gauss-', num2str(X_Min), '-', num2str(X_Max), '_' ];

	else
		error('Not_Implemented_Error');
	end

end

function [D, E_Log_P, Str] = Choose_Prior(Para, Opt, Str)

	EPS = Para.EPS;

	epsilon = Para.epsilon;

	rho = Para.rho;

	if strcmp(Opt, 'Rad')
		D = @ (w) D_Rad(EPS, w);
		E_Log_P = @ (w) E_Log_P_Rad(EPS, w);
		Str = [ Str, 'Rad', '_' ];
	elseif strcmp(Opt, '2Points')
		D = @ (w) D_2Points(EPS, epsilon, w);
		E_Log_P = @ (w) E_Log_P_2Points(EPS, epsilon, w);
		Str = [ Str, '2Points-', num2str(epsilon), '_' ];
	elseif strcmp(Opt, 'Gauss_Ber')
		D = @ (w) D_Gauss_Ber(EPS, rho, w);
		E_Log_P = @ (w) E_Log_P_Gauss_Ber(EPS, rho, w);
		Str = [ Str, 'Gauss_Ber-', num2str(rho), '_' ];
	else
		error('Not_Implemented_Error');
	end

end

function Res = D_Rad(EPS, w)

	w = max( EPS, min(1 - EPS, w) );

	h = w / max(EPS, 1 - w);

	Res = integral( @ (x) ...
		tanh( sqrt(h) * x + h ) .* normpdf(x), ...
	- Inf, Inf );

end

function Res = E_Log_P_Rad(EPS, w)

	w = max( EPS, min(1 - EPS, w) );

	h = w / max(EPS, 1 - w);

	% 定义被积函数
	% 使用 log( cosh(x) ) = abs(x) + log( ( 1 + exp( - 2 * abs(x) ) ) / 2 ) 避免溢出
	Log_Cosh = @ (x) abs(x) + log1p( exp( - 2 * abs(x) ) ) - log(2);

	% 计算期望项
	% Func = @ (x) log( cosh( h + sqrt(h) * x ) ) .* normpdf(x);

	Func = @ (x) Log_Cosh( h + sqrt(h) * x ) .* normpdf(x);

	Res = integral(Func, - Inf, Inf) - h / 2;

end

function Res = D_2Points(EPS, epsilon, w)

	w = max( EPS, min(1 - EPS, w) );

	h = w / max(EPS, 1 - w);

	Func = @ (x) 1 ./ ( ...
		epsilon^(2) + ( 1 - epsilon^(2) ) * exp( ...
			- h / ( 2 * epsilon^(2) ) ...
			- sqrt(h) / epsilon * x ...
		) ...
	);

	Res = epsilon^(2) * integral( @ (x) ...
		Func(x) .* normpdf(x), ...
	- Inf, Inf );

end

function Res = E_Log_P_2Points(EPS, epsilon, w)

	w = max( EPS, min(1 - EPS, w) );

	h = w / max(EPS, 1 - w);

	e2 = epsilon^(2);

	% 预计算解析公式中的常数
	Log_A = log(1 - e2);
	Log_B = log(e2);

	% 数值稳定地计算log( exp(Log_A) + exp(x) )
	% 原理: log( e^a + e^b ) = max(a, b) + log( 1 + exp( - abs(a - b) ) )
	Log_Sum_Exp = @ (x) max(Log_A, x) + log1p( exp( - abs(Log_A - x) ) );

	% I_0 = @ (x) log( ...
	% 	1 - e2 + e2 * exp( ...
	% 		- h / (2 * e2) + sqrt(h) / epsilon * x ...
	% 	) ...
	% ) .* normpdf(x);

	% I_Spike = @ (x) log( ...
	% 	1 - e2 + e2 * exp( ...
	% 		h / (2 * e2) + sqrt(h) / epsilon * x ...
	% 	) ...
	% ) .* normpdf(x);

	I_0 = @ (x) Log_Sum_Exp( ...
		Log_B - h / (2 * e2) + sqrt(h) / epsilon * x ...
	) .* normpdf(x);

	I_Spike = @ (x) Log_Sum_Exp( ...
		Log_B + h / (2 * e2) + sqrt(h) / epsilon * x ...
	) .* normpdf(x);

	Res = (1 - e2) * integral( I_0, - Inf, Inf ) ...
	+ e2 * integral( I_Spike, - Inf, Inf );

end

function Res = D_Gauss_Ber(EPS, rho, w)

	w = max( EPS, min(1 - EPS, w) );

	h = w / max(EPS, 1 - w);

	v = 1 / rho;

	Sigma2 = 1 / h;

	Gamma = v / (v + Sigma2);

	Func = @ (x) 1 ./ ( ...
		1 + (1 - rho) / rho * sqrt( (v + Sigma2) / Sigma2 ) * exp( ...
			- v / ( 2 * Sigma2 * (v + Sigma2) ) * x.^(2) ...
		) ...
	);

	Res = rho * Gamma^(2) * integral( @ (x) ...
		x.^(2) .* Func(x) .* normpdf( x, 0, sqrt(v + Sigma2) ), ...
	- Inf, Inf );

end

function Res = E_Log_P_Gauss_Ber(EPS, rho, w)

	w = max( EPS, min(1 - EPS, w) );

	h = w / max(EPS, 1 - w);

	% 预计算解析公式中的常数
	Log_A = log(1 - rho);
	Log_B = log(rho) + 1 / 2 * log( rho / ( rho + h) );

	% 解析简化后的单重积分 (Simplified Single Integral) ---
	% 我们推导出的变量合并: Xi = h * u0 + sqrt(h) * z
	Sigma_Xi = sqrt( h * (h + rho) / rho );

	% 数值稳定地计算log( exp(Log_A) + exp(x) )
	% 原理: log( e^a + e^b ) = max(a, b) + log( 1 + exp( - abs(a - b) ) )
	Log_Sum_Exp = @ (x) max(Log_A, x) + log1p( exp( - abs(Log_A - x) ) );

	% I_0 = @(z) log( ...
	% 	1 - rho ...
	% 	+ rho * sqrt( rho / ( rho + h) ) * exp( ...
	% 		h / ( 2 * (rho + h) ) * z.^(2) ...
	% 	) ...
	% ) .* normpdf(z);

	% I_Gauss = @ (Xi) log( ...
	% 	1 - rho ...
	% 	+ rho * sqrt( rho / ( rho + h) ) * exp( ...
	% 		1 / ( 2 * (rho + h) ) * Xi.^(2) ...
	% 	) ...
	% ) .* normpdf(Xi, 0, Sigma_Xi);

	% 使用数值稳定的logsumexp逻辑
	I_0 = @(z) Log_Sum_Exp( ...
		Log_B + h / ( 2 * (rho + h) ) * z.^(2) ...
	) .* normpdf(z);

	I_Gauss = @ (Xi) Log_Sum_Exp( ...
		Log_B + 1 / ( 2 * (rho + h) ) * Xi.^(2) ...
	) .* normpdf(Xi, 0, Sigma_Xi);

	% 执行单重积分
	Res = (1 - rho) * integral(I_0, - Inf, Inf) ...
	+ rho *integral(I_Gauss, - Inf, Inf);

end

function Res = Phi_1(Para, Theta, x)
	
	Delta = Para.Delta;

	Rho = Para.Rho;
	H = Para.H;

	De1 = 1 - Theta^(2) * Delta * pi^(2) * x .* H(x).^(2) + ...
	Theta^(2) * Delta * pi^(2) * x .* Rho(x).^(2) - ...
	Theta^(2) * (1 - Delta) * pi * H(x);

	De2 = 2 * Theta^(2) * Delta * pi^(2) * x .* H(x) .* Rho(x) + ...
	Theta^(2) * (1 - Delta) * pi * Rho(x);

	Res = ( ...
		1 + Theta^(2) * Delta * pi^(2) * x .* ( H(x).^(2) + Rho(x).^(2) ) ...
	) ./ ( ...
		De1.^(2) + De2.^(2) ...
	);

end

function Res = Phi_2(Para, Theta, x)
	
	Delta = Para.Delta;

	Rho = Para.Rho;
	H = Para.H;

	De1 = 1 - Theta^(2) * Delta * pi^(2) * x .* H(x).^(2) + ...
	Theta^(2) * Delta * pi^(2) * x .* Rho(x).^(2) - ...
	Theta^(2) * (1 - Delta) * pi * H(x);

	De2 = 2 * Theta^(2) * Delta * pi^(2) * x .* H(x) .* Rho(x) + ...
	Theta^(2) * (1 - Delta) * pi * Rho(x);

	Res = ( ...
		Delta * x + Theta^(2) * Delta^(2) * pi^(2) * x.^(2) .* ( H(x).^(2) + Rho(x).^(2) ) + ...
		2 * Theta^(2) * Delta * (1 - Delta) * pi * x .* H(x) + ...
		(1 - Delta)^(2) * Theta^(2) ...
	) ./ ( ...
		x .* ( De1.^(2) + De2.^(2) ) ...
	);

end

function Res = Phi_3(Para, Theta, x)
	
	Delta = Para.Delta;

	Rho = Para.Rho;
	H = Para.H;

	De1 = 1 - Theta^(2) * Delta * pi^(2) * x .* H(x).^(2) + ...
	Theta^(2) * Delta * pi^(2) * x .* Rho(x).^(2) - ...
	Theta^(2) * (1 - Delta) * pi * H(x);

	De2 = 2 * Theta^(2) * Delta * pi^(2) * x .* H(x) .* Rho(x) + ...
	Theta^(2) * (1 - Delta) * pi * Rho(x);

	Res = ( ...
		Theta * ( 1 - Delta + 2 * Delta * pi * x .* H(x) ) ...
	) ./ ( ...
		De1.^(2) + De2.^(2) ...
	);

end

function Output = Compute_Residual(Para, Theta, Input)

	EPS = Para.EPS;

	Delta = Para.Delta;

	X_Min = Para.X_Min;
	X_Max = Para.X_Max;
	Rho = Para.Rho;
	S0 = Para.S0;

	D_U = Para.D_U;
	D_V = Para.D_V;

	W_U = Input(1);
	W_V = Input(2);

	W_U = max( EPS, min(1 - EPS, W_U) );
	W_V = max( EPS, min(1 - EPS, W_V) );

	HM_U = W_U / max(EPS, 1 - W_U);
	HM_V = W_V / max(EPS, 1 - W_V);

	HM_U = max(EPS, HM_U);
	HM_V = max(EPS, HM_V);

	M_U = D_U(W_U);
	M_V = D_V(W_V);

	MMSE_U = 1 - M_U;
	MMSE_V = 1 - M_V;

	MMSE_U = max(EPS, MMSE_U);
	MMSE_V = max(EPS, MMSE_V);

	Rho_U = 1 / max(EPS, MMSE_U) - 1 / max(EPS, 1 - W_U);
	Rho_U = max(EPS, Rho_U);

	Rho_V = 1 / max(EPS, MMSE_V) - 1 / max(EPS, 1 - W_V);
	Rho_V = max(EPS, Rho_V);

	% 矩阵Denoiser对应的P*, Q*
	De = @ (x) ( Rho_U * Phi_1(Para, Theta, x) + 1 ) .* ...
	( Rho_V * Phi_2(Para, Theta, x) + Delta ) .* x - ...
	Rho_U * Rho_V * ( Phi_3(Para, Theta, x) ).^(2);

	Ps = @ (x) x .* ( Rho_V * Phi_2(Para, Theta, x) + Delta ) ./ De(x);
	Qs = @ (x) Delta * x .* ( Rho_U * Phi_1(Para, Theta, x) + 1 ) ./ De(x);

	E_Ps = integral( @ (x) Rho(x) .* Ps(x), X_Min, X_Max );
	E_Ps = max(EPS, E_Ps);

	E_Qs = Delta * integral( @ (x) Rho(x) .* Qs(x), X_Min, X_Max ) + ...
	(1 - Delta) * Delta / ( ...
		Rho_V * Delta / ( 1 - Theta^(2) * (1 - Delta) * S0 ) + Delta ...
	);
	E_Qs = max(EPS, E_Qs);

	New_W_U = 1 - (1 - E_Ps) / max(EPS, Rho_U * E_Ps);
	New_W_V = 1 - (1 - E_Qs) / max(EPS, Rho_V * E_Qs);

	Res_1 = 1 - (1 - E_Ps) / max(EPS, Rho_U * E_Ps) - W_U;
	Res_2 = 1 - (1 - E_Qs) / max(EPS, Rho_V * E_Qs) - W_V;

	Output = [Res_1; Res_2];

end

function Index_Array = Find_Position(Residual)

	% 假设Residual是你的残差矩阵[rows_grid, cols_grid]
	[M, N] = size(Residual);

	% 初始化一个逻辑矩阵，默认全为true(都是潜在极小值)
	Is_Min = true(M, N);

	% 定义8个位移方向[d_row, d_col]
	Directions = [ ...
		- 1, - 1; ...
		- 1, 0; ...
		- 1, 1; ...
		0, - 1; ...
		0, 1; ...
		1, - 1; ...
		1, 0; ...
		1, 1 ...
	];

	for i = 1 : size(Directions, 1)
		D_R = Directions(i, 1);
		D_C = Directions(i, 2);

		% 构造一个平移后的副本Result
		Result = NaN(M, N);

		% 计算源区域(对应的原位置)
		Row_Src = max(1, 1 - D_R) : min(M, M - D_R);
		Col_Src = max(1, 1 - D_C) : min(N, N - D_C);

		% 计算目标区域(Shift后的位置)
		Row_Dst = max(1, 1 + D_R) : min(M, M + D_R);
		Col_Dst = max(1, 1 + D_C) : min(N, N + D_C);

		% 执行平移
		Result(Row_Dst, Col_Dst) = Residual(Row_Src, Col_Src);

		% 核心对比逻辑
		% 只有当原点Residual小于等于平移过来的邻居时，它才保持为true
		% isnan 用于处理边界，如果邻居超出边界，我们认为边界外是“无穷大”，不影响判断
		Is_Min = Is_Min & ( Residual <= Result | isnan(Result) );
	end

	% 提取结果
	% 除了是局部极小，残差还必须足够小(比如<1e-4)才是方程的真正解
	[Row_Array, Col_Array] = find(Is_Min);

	Index_Array = [Row_Array'; Col_Array'];

end