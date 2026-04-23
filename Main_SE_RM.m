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
Opt_U = 'Rad';
% Opt_U = '2Points';
Para.Opt_U = Opt_U;

epsilon = 1 / 12;
% epsilon = 1 / 8;
% epsilon = 1 / 4;
Para.epsilon = epsilon;

rho = 0.4;
Para.rho = rho;

Opt_V = 'Rad';
% Opt_V = 'Gauss_Ber';
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

% Theta_Interval = 0.001;
Theta_Interval = 0.01;

Theta_Array = [ 0.2 : Theta_Interval : 2 ];
% Theta_Array = [ 0.28 : Theta_Interval : 0.34 ];
Str = [ Str, 'T-', num2str(Theta_Array(1)), '-', num2str(Theta_Interval), '-', num2str(Theta_Array(end)) ];

Num_Theta = length(Theta_Array);

SE_M_U_fM_Array = zeros(1, Num_Theta);
SE_M_V_fM_Array = zeros(1, Num_Theta);

SE_M_U_fW_Array = zeros(1, Num_Theta);
SE_M_V_fW_Array = zeros(1, Num_Theta);

SE_W_U_Array = zeros(1, Num_Theta);
SE_W_V_Array = zeros(1, Num_Theta);

PCA_U_Array = zeros(1, Num_Theta);
PCA_V_Array = zeros(1, Num_Theta);

RM_M_U_fM_Array = zeros(1, Num_Theta);
RM_M_V_fM_Array = zeros(1, Num_Theta);

RM_M_U_fW_Array = zeros(1, Num_Theta);
RM_M_V_fW_Array = zeros(1, Num_Theta);

RM_W_U_Array = zeros(1, Num_Theta);
RM_W_V_Array = zeros(1, Num_Theta);

Conv = 1e-6;
Para.Conv = Conv;

Iter = 200; % OAMP-SE迭代次数
Para.Iter = Iter;

for t = 1 : Num_Theta

	Theta = Theta_Array(t);
	Lambda = Theta / sqrt(Alpha);

	% 通过Outlier方程得到J_L, J_R
	[J1_L, J1_R, J2_L, J2_R, ~, ~] = Compute_Outlier_Masses(Para, Theta);

	J1_L_C = max( 0, min( 1, real(J1_L) ) );
	J1_R_C = max( 0, min( 1, real(J1_R) ) );
	J2_L_C = max( 0, min( 1, real(J2_L) ) );
	J2_R_C = max( 0, min( 1, real(J2_R) ) );

	% 标准PCA: 只用右侧Outlier的质量
	PCA_U_Array(t) = J1_R_C;
	PCA_V_Array(t) = J2_R_C;

	% OAMP初始化: optimal spectral mass
	W_U_Init = min( 1, J1_L_C + J1_R_C );
	W_V_Init = min( 1, J2_L_C + J2_R_C );

	% OAMP-SE递推Iter步骤
	% [SE_M_U, SE_M_V, SE_W_U, SE_W_V] = OAMP_SE(Para, Theta, W_U_Init, W_V_Init);
	[SE_M_U, SE_M_V, SE_W_U, SE_W_V] = OAMP_SE(Para, Theta, 0, 0);

	% 最终性能: Squared Cosine Similarity = 1 - MMSE(w_final) = D(w_final)
	SE_M_U_fM_Array(t) = SE_M_U;
	SE_M_V_fM_Array(t) = SE_M_V;

	SE_W_U_Array(t) = SE_W_U;
	SE_W_V_Array(t) = SE_W_V;

	SE_M_U_fW_Array(t) = max( EPS, min( 1 - EPS, D_U(SE_W_U) ) );
	SE_M_V_fW_Array(t) = max( EPS, min( 1 - EPS, D_V(SE_W_V) ) );

	% RM递推Iter步骤
	% 定义R_U和R_V的匿名函数包装
	R_U = @ (a1M_U, Rho_Va1) Solve_U(Para, Lambda, a1M_U, Rho_Va1);
	R_V = @ (a1M_V, Rho_Ua1) Solve_V(Para, Lambda, a1M_V, Rho_Ua1);

	% [RM_M_U, RM_M_V, RM_W_U, RM_W_V] = RM_Iter(Para, R_U, R_V, W_U_Init, W_V_Init);
	[RM_M_U, RM_M_V, RM_W_U, RM_W_V] = RM_Iter(Para, R_U, R_V, 0, 0);

	RM_M_U_fM_Array(t) = RM_M_U;
	RM_M_V_fM_Array(t) = RM_M_V;

	RM_W_U_Array(t) = RM_W_U;
	RM_W_V_Array(t) = RM_W_V;

	RM_M_U_fW_Array(t) = max( EPS, min( 1 - EPS, D_U(RM_W_U) ) );
	RM_M_V_fW_Array(t) = max( EPS, min( 1 - EPS, D_V(RM_W_V) ) );

end

for t = 1 : Num_Theta

	Theta = Theta_Array(t);

	if 0 == mod(t, 20) || 1 == t || t == Num_Theta

		fprintf('Theta: %d\n', Theta);
		fprintf('[PCA], U: %d, V: %d\n', PCA_U_Array(t), PCA_V_Array(t));
		fprintf('[SE], W_U: %d, U-fM: %d, U-fW: %d, W_V: %d, V-fM: %d, V-fW: %d\n', ...
			SE_W_U_Array(t), SE_M_U_fM_Array(t), SE_M_U_fW_Array(t), ...
			SE_W_V_Array(t), SE_M_V_fM_Array(t), SE_M_V_fW_Array(t) ...
		);
		fprintf('[RM], W_U: %d, U-fM: %d, U-fW: %d, W_V: %d, V-fM: %d, V-fW: %d\n', ...
			RM_W_U_Array(t), RM_M_U_fM_Array(t), RM_M_U_fW_Array(t), ...
			RM_W_V_Array(t), RM_M_V_fM_Array(t), RM_M_V_fW_Array(t) ...
		);

	end

end

figure( 'Position', [ 100, 100, 1200, 500 ] );

% U-direction
subplot(1, 2, 1);
hold on;
grid on;
box on;
plot( Theta_Array, PCA_U_Array, 'g:', 'LineWidth', 2.0, 'DisplayName', 'PCA-U' );
plot( Theta_Array, SE_M_U_fM_Array, 'bsquare', 'LineWidth', 0.5, 'DisplayName', 'SE-U-fM' );
plot( Theta_Array, SE_M_U_fW_Array, 'bx', 'LineWidth', 0.1, 'DisplayName', 'SE-U-fW' );
plot( Theta_Array, RM_M_U_fM_Array, 'ro', 'LineWidth', 0.5, 'DisplayName', 'RM-U-fM' );
plot( Theta_Array, RM_M_U_fW_Array, 'k+', 'LineWidth', 0.1, 'DisplayName', 'RM-U-fW' );
xlabel( 'Signal Strength \Theta' );
ylabel( 'Squared Cosine Similarity' );
title( sprintf( 'U-direction (\\Delta=%d)', Delta ) );
ylim( [ - 0.05, 1.05 ] );
legend( 'Location', 'northwest' );

% V-direction
subplot(1, 2, 2);
hold on;
grid on;
box on;
plot( Theta_Array, PCA_V_Array, 'g:', 'LineWidth', 2.0, 'DisplayName', 'PCA-V' );
plot( Theta_Array, SE_M_V_fM_Array, 'bsquare', 'LineWidth', 0.5, 'DisplayName', 'SE-V-fM' );
plot( Theta_Array, SE_M_V_fW_Array, 'bx', 'LineWidth', 0.1, 'DisplayName', 'SE-V-fW' );
plot( Theta_Array, RM_M_V_fM_Array, 'ro', 'LineWidth', 0.5, 'DisplayName', 'RM-V-fM' );
plot( Theta_Array, RM_M_V_fW_Array, 'k+', 'LineWidth', 0.1, 'DisplayName', 'RM-V-fW' );
xlabel( 'Signal Strength \Theta' );
ylabel( 'Squared Cosine Similarity' );
title( sprintf( 'V-direction (\\Delta=%d)', Delta ) );
ylim( [ - 0.05, 1.05 ] );
legend( 'Location', 'northwest' );
saveas( gcf, [ Str, '.fig' ] );

Ans_U = [ Theta_Array; SE_W_U_Array; SE_M_U_fM_Array; SE_M_U_fW_Array; RM_W_U_Array; RM_M_U_fM_Array; RM_M_U_fW_Array ];
Ans_V = [ Theta_Array; SE_W_V_Array; SE_M_V_fM_Array; SE_M_V_fW_Array; RM_W_V_Array; RM_M_V_fM_Array; RM_M_V_fW_Array ];

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

function [J1_L, J1_R, J2_L, J2_R, Has_LO, Has_RO] = Compute_Outlier_Masses(Para, Theta)

	Delta = Para.Delta;

	X_Min = Para.X_Min;
	X_Max = Para.X_Max;

	Mid = (X_Min + X_Max) / 2;

	S1 = @ (x) 2 * ( x - Mid + sqrt( (x - X_Min) .* (x - X_Max) ) );
	S1P = @ (x) 2 + 2 * (x - Mid) ./ sqrt( (x - X_Min) .* (x - X_Max) );
	HS1 = @ (x) Delta * S1(x) + (1 - Delta) ./ x;

	S2 = @ (x) 2 * ( x - Mid - sqrt( (x - X_Min) .* (x - X_Max) ) );
	S2P = @ (x) 2 - 2 * (x - Mid) ./ sqrt( (x - X_Min) .* (x - X_Max) );
	HS2 = @ (x) Delta * S2(x) + (1 - Delta) ./ x;

	ELO = @ (x) 1 - Theta^(2) * ( Delta * x .* S1(x).^(2) + (1 - Delta) * S1(x) );
	ERO = @ (x) 1 - Theta^(2) * ( Delta * x .* S2(x).^(2) + (1 - Delta) * S2(x) );

	ELOP = @ (x) - Theta^(2) .* ( Delta * S1(x).^(2) + 2 * Delta * x .* S1(x) .* S1P(x) + (1 - Delta) * S1P(x) );
	EROP = @ (x) - Theta^(2) .* ( Delta * S2(x).^(2) + 2 * Delta * x .* S2(x) .* S2P(x) + (1 - Delta) * S2P(x) );

	Opts = optimset('Display', 'off');

	J1_L = 0;
	J2_L = 0; 
	J1_R = 0;
	J2_R = 0;

	Has_LO = false; 
	Has_RO = false;

	% 左Outlier
	try
		LO = fzero(ELO, [1e-6, X_Min - 1e-6], Opts);
		De = ELOP(LO);
		if isfinite(De) && abs(De) > 1e-14
			J1_L = max( real(S1(LO) / De), 0 );
			J2_L = max( real(HS1(LO) / De), 0 );
			Has_LO = (J1_L > 1e-12) || (J2_L > 1e-12);
		end
	catch
	end

	% 右Outlier
	try
		RO = fzero(ERO, [X_Max + 1e-6, 15], Opts);
		De = EROP(RO);
		if isfinite(De) && abs(De) > 1e-14
			J1_R = max( real(S2(RO) / De), 0 );
			J2_R = max( real(HS2(RO) / De), 0 );
			Has_RO = (J1_R > 1e-12) || (J2_R > 1e-12);
		end
	catch
	end

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

function [M_U, M_V, W_U, W_V] = OAMP_SE(Para, Theta, W_U_Init, W_V_Init)

	EPS = Para.EPS;

	Delta = Para.Delta;

	X_Min = Para.X_Min;
	X_Max = Para.X_Max;
	Rho = Para.Rho;
	S0 = Para.S0;

	D_U = Para.D_U;
	D_V = Para.D_V;

	Conv = Para.Conv;
	Iter = Para.Iter;

	i = 1;
	Crit = 10;

	M_U = 10;
	M_V = 10;

	W_U = W_U_Init;
	W_V = W_V_Init;

	while (Crit > Conv && i < Iter)

		W_U = max( EPS, min(1 - EPS, W_U) );
		W_V = max( EPS, min(1 - EPS, W_V) );

		HM_U = W_U / max(EPS, 1 - W_U);
		HM_V = W_V / max(EPS, 1 - W_V);

		HM_U = max(EPS, HM_U);
		HM_V = max(EPS, HM_V);

		M_U_Old = M_U;
		M_V_Old = M_V;

		M_U = D_U(W_U);
		M_V = D_V(W_V);

		Crit = ( abs(M_U_Old - M_U) + abs(M_V_Old - M_V) );

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

		W_U = 1 - (1 - E_Ps) / max(EPS, Rho_U * E_Ps);

		W_U = max( EPS, min(1 - EPS, W_U) );

		W_V = 1 - (1 - E_Qs) / max(EPS, Rho_V * E_Qs);

		W_V = max( EPS, min(1 - EPS, W_V) );

		i = i + 1;

	end

end

function Res = Solve_U(Para, Lambda, a1M_U, Rho_Va1)

	Alpha = Para.Alpha;

	X_Min = Para.X_Min;
	X_Max = Para.X_Max;
	Rho = Para.Rho;
	H = Para.H;
	S = Para.S;

	Precision = 100000;
	X_Array = linspace(X_Min, X_Max, Precision);

	J_UU = @ (x) Lambda^(2) * pi^(2) * x .* ( ( Rho(x) ).^(2) + ( H(x) ).^(2) ) + ...
	(Alpha - 1) * Lambda^(2) * 2 * pi * H(x) + ...
	(Alpha - 1)^(2) * Lambda^(2) * 1 ./ x;

	J_VV = @ (x) Lambda^(2) * pi^(2) * x .* ( ( Rho(x) ).^(2) + ( H(x) ).^(2) );

	J_UV = @ (x) Lambda^(2) * 4 * pi^(2) * x .* ( H(x) ).^(2) + ...
	(Alpha - 1) * Lambda^(2) * 4 * pi * H(x) + ...
	(Alpha - 1)^(2) * Lambda^(2) * 1 ./ x;

	J_U = @ (x) - J_UU(x) + J_UV(x) ./ (Rho_Va1 + J_VV(x));

	Func = @ (z) integral( @ (x) Rho(x) ./ (z - J_U(x)), X_Min, X_Max ) - a1M_U;

	J_U_X = J_U(X_Array);
	z_Max = max(J_U_X);
	z1 = z_Max + 1e-5;
	z2 = z_Max + 1e30;

	if Func(z1) < 0

		Res = z1 - 1 / a1M_U;

	elseif Func(z2) > 0

		Res = z2 - 1 / a1M_U;

	else

		Res = Bisection(Func, z1, z2, 200) - 1 / a1M_U;

	end

end

function Res = Solve_V(Para, Lambda, a1M_V, Rho_Ua1)

	Alpha = Para.Alpha;

	X_Min = Para.X_Min;
	X_Max = Para.X_Max;
	Rho = Para.Rho;
	H = Para.H;
	S0 = Para.S0;

	Precision = 100000;
	X_Array = linspace(X_Min, X_Max, Precision);

	J_UU = @ (x) Lambda^(2) * pi^(2) * x .* ( ( Rho(x) ).^(2) + ( H(x) ).^(2) ) + ...
	(Alpha - 1) * Lambda^(2) * 2 * pi * H(x) + ...
	(Alpha - 1)^(2) * Lambda^(2) * 1 ./ x;

	J_VV = @ (x) Lambda^(2) * pi^(2) * x .* ( ( Rho(x) ).^(2) + ( H(x) ).^(2) );

	J_UV = @ (x) Lambda^(2) * 4 * pi^(2) * x .* ( H(x) ).^(2) + ...
	(Alpha - 1) * Lambda^(2) * 4 * pi * H(x) + ...
	(Alpha - 1)^(2) * Lambda^(2) * 1 ./ x;

	a_V = - (Alpha - 1) * Lambda^(2) * S0;

	J_V = @ (x) - J_VV(x) + J_UV(x) ./ (Rho_Ua1 + J_UU(x));
	J_V0 = - a_V;

	Func = @ (z) (1 / Alpha) * integral( @ (x) Rho(x) ./ (z - J_V(x)), X_Min, X_Max ) + ...
	(1 - 1 / Alpha) * 1 / (z - J_V0) - a1M_V;

	J_V_X = J_V(X_Array);
	z_Max = max(J_V_X);
	z_Max = max(z_Max, J_V0);
	z1 = z_Max + 1e-5;
	z2 = z_Max + 1e30;

	if Func(z1) < 0

		Res = z1 - 1 / a1M_V;

	elseif Func(z2) > 0

		Res = z2 - 1 / a1M_V;

	else

		Res = Bisection(Func, z1, z2, 200) - 1 / a1M_V;

	end

end

function Mid = Bisection(Func, x1, x2, Precision)

	for i = 1 : Precision

		Mid = (x1 + x2) / 2;

		if Func(Mid) < 0
			x2 = Mid;
		else
			x1 = Mid;
		end

	end

	Mid = (x1 + x2) / 2;

end

function [M_U, M_V, W_U, W_V] = RM_Iter(Para, R_U, R_V, W_U_Init, W_V_Init)

	EPS = Para.EPS;

	Rho = Para.Rho;
	H = Para.H;
	S = Para.S;

	D_U = Para.D_U;
	D_V = Para.D_V;

	Conv = Para.Conv;
	Iter = Para.Iter;

	i = 1;
	Crit = 10;

	M_U = 10;
	M_V = 10;

	W_U = W_U_Init;
	W_V = W_V_Init;

	while (Crit > Conv && i < Iter)

		W_U = max( EPS, min(1 - EPS, W_U) );
		W_V = max( EPS, min(1 - EPS, W_V) );

		HM_U = W_U / max(EPS, 1 - W_U);
		HM_V = W_V / max(EPS, 1 - W_V);

		HM_U = max(EPS, HM_U);
		HM_V = max(EPS, HM_V);

		M_U_Old = M_U;
		M_V_Old = M_V;

		M_U = D_U(W_U);
		M_V = D_V(W_V);

		Crit = ( abs(M_U_Old - M_U) + abs(M_V_Old - M_V) );

		MMSE_U = 1 - M_U;
		MMSE_V = 1 - M_V;

		MMSE_U = max(EPS, MMSE_U);
		MMSE_V = max(EPS, MMSE_V);

		Rho_U = 1 / max(EPS, MMSE_U) - 1 / max(EPS, 1 - W_U);

		Rho_U = max(EPS, Rho_U);

		Rho_V = 1 / max(EPS, MMSE_V) - 1 / max(EPS, 1 - W_V);

		Rho_V = max(EPS, Rho_V);

		Rho_Ua1 = Rho_U + 1;
		Rho_Va1 = Rho_V + 1;

		HM_U = - R_U(MMSE_U, Rho_Va1);
		HM_V = - R_V(MMSE_V, Rho_Ua1);

		HM_U = max(EPS, HM_U);
		HM_V = max(EPS, HM_V);

		W_U = HM_U / (1 + HM_U);
		W_V = HM_V / (1 + HM_V);

		W_U = max( EPS, min(1 - EPS, W_U) );
		W_V = max( EPS, min(1 - EPS, W_V) );

		i = i + 1;

	end

end