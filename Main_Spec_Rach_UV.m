% =========================================================================
% OAMP vs PCA: Squared Cosine Similarity vs Theta (Theory + Simulation)
%
% U方向采用两点稀疏先验:
% P_U = epsilon^2 * delta_{1/epsilon} + (1-epsilon^2) * delta_0。
% V方向采用Gauss-Bernoulli先验:
% P_V = rho * N(0,1/rho) + (1-rho) * delta_0。
% =========================================================================
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
Para.Spectrum = Spectrum;

Str = [ Str, Spectrum, '_' ];

Alpha = 2.0;
Para.Alpha = Alpha;

if Alpha < 1
	error('This implementation requires Alpha >= 1 so that M <= N.');
end

Delta = 1 / Alpha; % M / N
Para.Delta = Delta;

Opt_U = '2Points';
Opt_V = 'Gauss_Ber';
Para.Opt_U = Opt_U;
Para.Opt_V = Opt_V;

% epsilon = 1 / 4;
epsilon = 1 / 12;
rho = 0.4;

Str = [ Str, Opt_U, '_', num2str(epsilon), '_' ];

Str = [ Str, Opt_V, '_', num2str(rho), '_' ];

if epsilon <= 0 || epsilon >= 1
	error('epsilon must satisfy 0 < epsilon < 1.');
end

if rho <= 0 || rho >= 1
	error('rho must satisfy 0 < rho < 1.');
end

Para.epsilon = epsilon;
Para.rho = rho;

% 获取势函数相关参数
[X_Min, X_Max, Rho, H, S, S0, Str] = Choose_Potential(Para, Str);

Para.X_Min = X_Min;
Para.X_Max = X_Max;
Para.Rho = Rho;
Para.H = H;
Para.S = S;
Para.S0 = S0;

% 获取先验分布、标量去噪器和随机变量生成器
[D_U, DMMSE_Est_U, MMSE_Est_U, Generate_U, Mean_U, Moment_4_U, Str] = ...
	Choose_Prior(Para, Opt_U, Str);

[D_V, DMMSE_Est_V, MMSE_Est_V, Generate_V, Mean_V, Moment_4_V, Str] = ...
	Choose_Prior(Para, Opt_V, Str);

Para.D_U = D_U;
Para.D_V = D_V;
Para.DMMSE_Est_U = DMMSE_Est_U;
Para.DMMSE_Est_V = DMMSE_Est_V;
Para.MMSE_Est_U = MMSE_Est_U;
Para.MMSE_Est_V = MMSE_Est_V;
Para.Generate_U = Generate_U;
Para.Generate_V = Generate_V;
Para.Mean_U = Mean_U;
Para.Mean_V = Mean_V;
Para.Moment_4_U = Moment_4_U;
Para.Moment_4_V = Moment_4_V;

Generate_W = @ (M, N) Generate_Noise_Rad(M, N, X_Min, X_Max);
Para.Generate_W = Generate_W;

Str = [ Str, 'A_', num2str(Alpha), '_' ];

%% ===================== Part 1: SE vs PCA =====================
Theta_SE_Array = 0.2 : 0.01 : 2;
Num_Theta_SE = length(Theta_SE_Array);

Str = [ Str, 'TS_', num2str(Theta_SE_Array(1)), '_', num2str(Theta_SE_Array(end)), '_' ];

Iter_SE = 6;
Para.Iter_SE = Iter_SE;

SE_M_U_fW_Array = zeros(1, Num_Theta_SE);
SE_M_V_fW_Array = zeros(1, Num_Theta_SE);

PCA_U_Array = zeros(1, Num_Theta_SE);
PCA_V_Array = zeros(1, Num_Theta_SE);

fprintf('========== Part 1: 扫描Theta (SE vs PCA) ==========\n');

for t = 1 : Num_Theta_SE

	Theta = Theta_SE_Array(t);

	% 通过Outlier方程得到J_L和J_R
	[J1_L, J1_R, J2_L, J2_R, ~, ~] = Compute_Outlier_Masses(Para, Theta);

	J1_L_C = max( 0, min( 1, real(J1_L) ) );
	J1_R_C = max( 0, min( 1, real(J1_R) ) );
	J2_L_C = max( 0, min( 1, real(J2_L) ) );
	J2_R_C = max( 0, min( 1, real(J2_R) ) );

	% 标准PCA: 只使用右侧Outlier质量
	PCA_U_Array(t) = J1_R_C;
	PCA_V_Array(t) = J2_R_C;

	% OAMP初始化: optimal spectral mass
	W_U_Init = min( 1, J1_L_C + J1_R_C );
	W_V_Init = min( 1, J2_L_C + J2_R_C );

	% 理论SE固定递推Iter_SE步
	[M_U, M_V, ~, ~] = OAMP_SE(Para, Theta, W_U_Init, W_V_Init);

	SE_M_U_fW_Array(t) = max( 0, M_U );
	SE_M_V_fW_Array(t) = max( 0, M_V );

	if 0 == mod(t, 20) || 1 == t || t == Num_Theta_SE
		fprintf(' [SE] Theta: %.2f, OAMP-U: %.4f, PCA-U: %.4f, OAMP-V: %.4f, PCA-V: %.4f\n', ...
			Theta, SE_M_U_fW_Array(t), PCA_U_Array(t), SE_M_V_fW_Array(t), PCA_V_Array(t) ...
		);
	end

end

%% ===================== Part 2: OAMP仿真 =====================
% Theta_OAMP_Array = 0.2 : 0.1 : 2;
% Theta_OAMP_Array = 0.2 : 0.1 : 1.6;
% Theta_OAMP_Array = [0.4];
Theta_OAMP_Array = 0.3 : 0.2 : 2;

Num_Theta_OAMP = length(Theta_OAMP_Array);

Str = [ Str, 'TO_', num2str(Theta_OAMP_Array(1)), '_', num2str(Theta_OAMP_Array(end)), '_' ];

Iter_OAMP = 6;
Para.Iter_OAMP = Iter_OAMP;

% N = 4000;
N = 8000;
M = round(Delta * N);
% Num_Reps = 50;
Num_Reps = 100;

Str = [ Str, 'N_', num2str(N), '_', 'M_', num2str(M) ];

Para.N = N;
Para.M = M;

OAMP_sCos_U_Array = zeros(1, Num_Theta_OAMP);
OAMP_sCos_V_Array = zeros(1, Num_Theta_OAMP);

OAMP_M_U_fW_Array = zeros(1, Num_Theta_OAMP);
OAMP_M_V_fW_Array = zeros(1, Num_Theta_OAMP);

fprintf('\n========== Part 2: OAMP仿真 (按Theta) ==========\n');

for k = 1 : Num_Theta_OAMP

	Theta = Theta_OAMP_Array(k);

	fprintf('Simulation for Theta=%.2f (%d/%d)\n', Theta, k, Num_Theta_OAMP);

	% 先根据Outlier方程判断是否存在Outlier
	[J1_L, J1_R, J2_L, J2_R, Has_LO, Has_RO] = Compute_Outlier_Masses(Para, Theta);

	J1_L_C = max( 0, min( 1, real(J1_L) ) );
	J1_R_C = max( 0, min( 1, real(J1_R) ) );
	J2_L_C = max( 0, min( 1, real(J2_L) ) );
	J2_R_C = max( 0, min( 1, real(J2_R) ) );

	OAMP_sCos_U_Reps_Array = zeros(1, Num_Reps);
	OAMP_sCos_V_Reps_Array = zeros(1, Num_Reps);

	OAMP_SE_U_Reps_Array = zeros(1, Num_Reps);
	OAMP_SE_V_Reps_Array = zeros(1, Num_Reps);

	for Rep = 1 : Num_Reps

		fprintf(' Rep %d / %d...\n', Rep, Num_Reps);

		% 保持原始代码的随机生成顺序: U_0, V_0, Beta谱, 高斯旋转矩阵
		U_0 = Generate_U(M);
		V_0 = Generate_V(N);
		W = Generate_W(M, N);

		Y = Theta * U_0 * V_0' / sqrt(M * N) + W;

		% 必须使用Full SVD
		[U_Y, S_Y, V_Y] = svd(Y);

		Singular_Y = diag( S_Y(1 : M, 1 : M) );
		Eigen_YYT = Singular_Y.^(2);

		% OAMP初始化: optimal spectral mass
		W_U_Init = min( 1, J1_L_C + J1_R_C );
		W_V_Init = min( 1, J2_L_C + J2_R_C );

		Z_U = randn(M, 1);
		Z_V = randn(N, 1);

		U_t_Init = sqrt(W_U_Init) * U_0 + sqrt(1 - W_U_Init) * Z_U;

		V_t_Init = sqrt(W_V_Init) * V_0 + sqrt(1 - W_V_Init) * Z_V;

		U_t_Init = sqrt(M) * U_t_Init / norm(U_t_Init);
		V_t_Init = sqrt(N) * V_t_Init / norm(V_t_Init);

		Para.U_0 = U_0;
		Para.V_0 = V_0;
		Para.U_Y = U_Y;
		Para.S_Y = S_Y;
		Para.V_Y = V_Y;
		Para.Singular_Y = Singular_Y;
		Para.Eigen_YYT = Eigen_YYT;

		[sCos_U, sCos_V, SE_U, SE_V] = OAMP_Iter( ...
			Para, Theta, Has_LO, Has_RO, U_t_Init, V_t_Init, W_U_Init, W_V_Init ...
		);

		OAMP_sCos_U_Reps_Array(Rep) = sCos_U;
		OAMP_sCos_V_Reps_Array(Rep) = sCos_V;

		OAMP_SE_U_Reps_Array(Rep) = SE_U;
		OAMP_SE_V_Reps_Array(Rep) = SE_V;

	end

	OAMP_sCos_U_Array(k) = mean(OAMP_sCos_U_Reps_Array);
	OAMP_sCos_V_Array(k) = mean(OAMP_sCos_V_Reps_Array);

	OAMP_M_U_fW_Array(k) = mean(OAMP_SE_U_Reps_Array);
	OAMP_M_V_fW_Array(k) = mean(OAMP_SE_V_Reps_Array);

	fprintf(' -> avg sCos(U)=%.4f, SE(U)=%.4f, avg sCos(V)=%.4f, SE(V)=%.4f\n\n', ...
		OAMP_sCos_U_Array(k), OAMP_M_U_fW_Array(k), OAMP_sCos_V_Array(k), OAMP_M_V_fW_Array(k) ...
	);

end

warning('on', 'MATLAB:integral:NonFiniteValue');
warning('on', 'MATLAB:fzero:NAx');

%% ===================== Part 3: 画图 =====================
figure( 'Position', [ 100, 100, 1200, 500 ] );

% U-direction
subplot(1, 2, 1);
hold on;
grid on;
box on;
plot( Theta_SE_Array, SE_M_U_fW_Array, 'b-', 'LineWidth', 2.2, 'DisplayName', 'OAMP SE (U)' );
plot( Theta_SE_Array, PCA_U_Array, 'k:', 'LineWidth', 2.0, 'DisplayName', 'Standard PCA (U)' );
plot( Theta_OAMP_Array, OAMP_sCos_U_Array, 'ro', 'LineWidth', 1.5, 'MarkerSize', 6, ...
	'DisplayName', 'OAMP Simulation (U)' ...
);
xlabel( 'Signal Strength \Theta' );
ylabel( 'Squared Cosine Similarity' );
title( sprintf( 'U-direction (\\Delta=%.2f)', Delta ) );
ylim( [ - 0.05, 1.05 ] );
legend( 'Location', 'northwest' );

% V-direction
subplot(1, 2, 2);
hold on;
grid on;
box on;
plot( Theta_SE_Array, SE_M_V_fW_Array, 'b-', 'LineWidth', 2.2, 'DisplayName', 'OAMP SE (V)' );
plot( Theta_SE_Array, PCA_V_Array, 'k:', 'LineWidth', 2.0, 'DisplayName', 'Standard PCA (V)' );
plot( Theta_OAMP_Array, OAMP_sCos_V_Array, 'ro', 'LineWidth', 1.5, 'MarkerSize', 6, ...
	'DisplayName', 'OAMP Simulation (V)' ...
);
xlabel( 'Signal Strength \Theta' );
ylabel( 'Squared Cosine Similarity' );
title( sprintf( 'V-direction (\\Delta=%.2f)', Delta ) );
ylim( [ - 0.05, 1.05 ] );
legend( 'Location', 'northwest' );

sgtitle({ ...
	'OAMP vs Standard PCA: Squared Cosine Similarity vs \Theta', ...
	sprintf( [ ...
		'\\Delta=%.2f, T=%d, U: two-point (\\epsilon=%.4g), ', ...
		'V: Gauss-Bernoulli (\\rho=%.4g), Beta noise on [%.0f,%.0f]' ...
	], Delta, Iter_OAMP, epsilon, rho, X_Min, X_Max ...
	) ...
});
saveas( gcf, [ Str, '.fig' ] );

save([Str, '.mat'])

% 核心功能函数定义
function [X_Min, X_Max, Rho, H, S, S0, Str] = Choose_Potential(Para, Str)

	Spectrum = Para.Spectrum;

	if strcmp(Spectrum, 'Rad')

		X_Min = 1;
		X_Max = X_Min + 2;

		Mid = (X_Min + X_Max) / 2;

		Rho = @ (x) 2 / pi * sqrt( ...
			max( 0, (x - X_Min) .* (X_Max - x) ) ...
		);

		H = @ (x) 2 / pi * (x - Mid);
		S = @ (x) 2 * ( x - Mid + sqrt( (x - X_Min) .* (x - X_Max) ) );
		S0 = S(0);

		Str = [ Str, 'Rad-', num2str(X_Min), '-', num2str(X_Max), '_' ];

	else
		error('Not_Implemented_Error');
	end

end

function [D, DMMSE_Est, MMSE_Est, Generate, Mean, Moment_4, Str] = ...
	Choose_Prior(Para, Opt, Str)

	EPS = Para.EPS;
	epsilon = Para.epsilon;
	rho = Para.rho;

	if strcmp(Opt, 'Rad')
		D = @ (w) D_Rad(EPS, w);
		DMMSE_Est = @ (mu, w) DMMSE_Est_Rad(mu, EPS, w);
		MMSE_Est = @ (mu, w) MMSE_Est_Rad(mu, EPS, w);
		Generate = @ (Dim) Generate_Rad(Dim);
		Mean = 0;
		Moment_4 = 1;
		Str = [ Str, 'Rad', '_' ];
	elseif strcmp(Opt, '2Points')
		D = @ (w) D_2Points(EPS, epsilon, w);
		DMMSE_Est = @ (mu, w) DMMSE_Est_2Points(mu, EPS, epsilon, w);
		MMSE_Est = @ (mu, w) MMSE_Est_2Points(mu, EPS, epsilon, w);
		Generate = @ (Dim) Generate_2Points(Dim, epsilon);
		Mean = epsilon;
		Moment_4 = 1 / epsilon^(2);
		Str = [ Str, '2Points-', num2str(epsilon), '_' ];
	elseif strcmp(Opt, 'Gauss_Ber')
		D = @ (w) D_Gauss_Ber(EPS, rho, w);
		DMMSE_Est = @ (mu, w) DMMSE_Est_Gauss_Ber(mu, EPS, rho, w);
		MMSE_Est = @ (mu, w) MMSE_Est_Gauss_Ber(mu, EPS, rho, w);
		Generate = @ (Dim) Generate_Gauss_Ber(Dim, rho);
		Mean = 0;
		Moment_4 = 3 / rho;
		Str = [ Str, 'Gauss_Ber-', num2str(rho), '_' ];
	else
		error('Not_Implemented_Error');
	end

end

function Res = D_Rad(EPS, w)

	Res = 1 - MMSE_Rad(EPS, w);

end

function Res = D_Rad_Extr(EPS, w)

	w = max( EPS, min(1 - EPS, w) );

	h = w / max(EPS, 1 - w);

	Res = integral( @ (x) ...
		tanh( sqrt(h) * x + h ) .* normpdf(x), ...
	- Inf, Inf );

end

function Res = MMSE_Rad(~, w)

	if w >= 1
		Res = 0;
		return;
	end

	MMSE_Est = @ (x) tanh( x .* sqrt(w) ./ (1 - w) );

	I_Minus = @ (z) ( ...
		- 1 - MMSE_Est( - sqrt(w) + sqrt(1 - w) .* z ) ...
	).^(2) .* normpdf(z);

	I_Plus = @ (z) ( ...
		1 - MMSE_Est( sqrt(w) + sqrt(1 - w) .* z ) ...
	).^(2) .* normpdf(z);

	Res = 1 / 2 * integral(I_Minus, - Inf, Inf) + ...
		1 / 2 * integral(I_Plus, - Inf, Inf);

end

function Res = E_Z_MMSE_Est_Rad_Extr(EPS, w)

	w = max( EPS, min(1 - EPS, w) );

	D = D_Rad_Extr(EPS, w);
	MMSE = 1 - D;
	Res = sqrt( w / (1 - w) ) * MMSE;	

end

function Res = E_Z_MMSE_Est_Rad(w)

	if w >= 1 || w <= 0
		Res = 0;
		return;
	end

	MMSE_Est = @ (x) tanh( x .* sqrt(w) ./ (1 - w) );

	I_Plus = @ (z) z .* MMSE_Est( ...
		sqrt(w) + sqrt(1 - w) .* z ...
	) .* normpdf(z);

	I_Minus = @ (z) z .* MMSE_Est( ...
		- sqrt(w) + sqrt(1 - w) .* z ...
	) .* normpdf(z);

	Res = 1 / 2 * integral(I_Plus, - Inf, Inf) + ...
		1 / 2 * integral(I_Minus, - Inf, Inf);

end

function Res = DMMSE_Est_Rad(mu, ~, w)

	if w >= 1
		Res = mu;
		return;
	end

	if w <= 0
		Res = zeros(size(mu));
		return;
	end

	E_Z_Phi = E_Z_MMSE_Est_Rad(w);

	Phi = tanh( mu .* sqrt(w) ./ (1 - w) );

	Res = ( ...
		1 - sqrt( w / (1 - w) ) * E_Z_Phi ...
	)^(-1) * ( ...
		Phi - E_Z_Phi / sqrt(1 - w) .* mu ...
	);

end

function Res = MMSE_Est_Rad(mu, ~, w)

	Res = tanh( mu .* sqrt(w) ./ (1 - w) );

end

function Res = Generate_Rad(Dim)

	Res = sign( randn(Dim, 1) );

end

function Res = D_2Points_Extr(EPS, epsilon, w)

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

function Res = D_2Points(EPS, epsilon, w)

	if w <= 0
		Res = epsilon^(2);
		return;
	end

	if w >= 1
		Res = 1;
		return;
	end

	w = max( EPS, min(1 - EPS, w) );
	h = w / (1 - w);
	epsilon_2 = epsilon^(2);

	Log_Odds = @ (z) log( epsilon_2 / (1 - epsilon_2) ) + ...
		h / (2 * epsilon_2) + sqrt(h) / epsilon .* z;

	Post_Active = @ (z) 1 ./ ( 1 + exp( - Log_Odds(z) ) );

	Res = integral( ...
		@ (z) Post_Active(z) .* normpdf(z), - Inf, Inf ...
	);

	Res = max( 0, min( 1, real(Res) ) );

end

function Res = DMMSE_Est_2Points(mu, EPS, epsilon, w)

	if w <= 0
		Res = epsilon * ones(size(mu));
		return;
	end

	if w >= 1
		Res = mu;
		return;
	end

	w = max( EPS, min(1 - EPS, w) );

	D = D_2Points(EPS, epsilon, w);
	MMSE = max( 0, 1 - D );
	E_Z_Phi = sqrt( w / (1 - w) ) * MMSE;

	Phi = MMSE_Est_2Points(mu, EPS, epsilon, w);

	Res = ( ...
		1 - sqrt( w / (1 - w) ) * E_Z_Phi ...
	)^(-1) * ( ...
		Phi - E_Z_Phi / sqrt(1 - w) .* mu ...
	);

end

function Res = MMSE_Est_2Points(mu, EPS, epsilon, w)

	if w <= 0
		Res = epsilon * ones(size(mu));
		return;
	end

	if w >= 1
		Res = mu;
		return;
	end

	w = max( EPS, min(1 - EPS, w) );

	Log_Odds = - log( epsilon^(2) / ( 1 - epsilon^(2) ) ) - ...
		sqrt(w) / ( epsilon * (1 - w) ) .* mu + ...
		w / ( 2 * epsilon^(2) * (1 - w) );

	Res = 1 / epsilon ./ ( 1 + exp(Log_Odds) );

end

function Res = Generate_2Points(Dim, epsilon)

	Mask = rand(Dim, 1) < epsilon^(2);

	Res = zeros(Dim, 1);
	Res(Mask) = 1 / epsilon;

end

function Res = D_Gauss_Ber_Extr(EPS, rho, w)

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

function Res = D_Gauss_Ber(EPS, rho, w)

	if w <= 0
		Res = 0;
		return;
	end

	if w >= 1
		Res = 1;
		return;
	end

	w = max( EPS, min(1 - EPS, w) );
	A = w + rho * (1 - w);
	Scale_Active = sqrt(A / rho);

	Log_Odds_Active = @ (z) log( rho / (1 - rho) ) + ...
		1 / 2 * log( rho * (1 - w) / A ) + ...
		w / ( 2 * (1 - w) * A ) .* (Scale_Active .* z).^(2);

	Post_Active = @ (z) 1 ./ ( 1 + exp( - Log_Odds_Active(z) ) );

	Res = w / A * integral( ...
		@ (z) z.^(2) .* Post_Active(z) .* normpdf(z), - Inf, Inf ...
	);

	Res = max( 0, min( 1, real(Res) ) );

end

function Res = DMMSE_Est_Gauss_Ber(mu, EPS, rho, w)

	if w <= 0
		Res = zeros(size(mu));
		return;
	end

	if w >= 1
		Res = mu;
		return;
	end

	w = max( EPS, min(1 - EPS, w) );

	D = D_Gauss_Ber(EPS, rho, w);
	MMSE = max( 0, 1 - D );
	E_Z_Phi = sqrt( w / (1 - w) ) * MMSE;

	Phi = MMSE_Est_Gauss_Ber(mu, EPS, rho, w);

	Res = ( ...
		1 - sqrt( w / (1 - w) ) * E_Z_Phi ...
	)^(-1) * ( ...
		Phi - E_Z_Phi / sqrt(1 - w) .* mu ...
	);

end

function Res = MMSE_Est_Gauss_Ber(mu, EPS, rho, w)

	if w <= 0
		Res = zeros(size(mu));
		return;
	end

	if w >= 1
		Res = mu;
		return;
	end

	w = max( EPS, min(1 - EPS, w) );
	A = w + rho * (1 - w);

	Log_Odds = log( rho / (1 - rho) ) + ...
		1 / 2 * log( rho * (1 - w) / A ) + ...
		w / ( 2 * (1 - w) * A ) .* mu.^(2);

	Post_Active = 1 ./ ( 1 + exp( - Log_Odds ) );

	Res = sqrt(w) / A .* mu .* Post_Active;

end

function Res = Generate_Gauss_Ber(Dim, rho)

	Mask = rand(Dim, 1) < rho;

	Res = Mask .* randn(Dim, 1) / sqrt(rho);

end

function Res = Generate_Noise_Rad(M, N, X_Min, X_Max)

	Nalpha = 1.5;
	Nbeta = 1.5;
	Nscale = X_Max - X_Min;
	Nshift = X_Min;

	B_Vec = betarnd(Nalpha, Nbeta, M, 1);
	Eigen_WWT = Nscale * B_Vec + Nshift;
	Singular_W = sqrt(Eigen_WWT);

	A = randn(M, N);
	[U_W, ~, V_W] = svd(A);

	Res = U_W * [ diag(Singular_W), zeros(M, N - M) ] * V_W';

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

	ELO = @ (x) 1 - Theta^(2) * ( ...
		Delta * x .* S1(x).^(2) + (1 - Delta) * S1(x) ...
	);

	ERO = @ (x) 1 - Theta^(2) * ( ...
		Delta * x .* S2(x).^(2) + (1 - Delta) * S2(x) ...
	);

	ELOP = @ (x) - Theta^(2) .* ( ...
		Delta * S1(x).^(2) + 2 * Delta * x .* S1(x) .* S1P(x) + ...
		(1 - Delta) * S1P(x) ...
	);

	EROP = @ (x) - Theta^(2) .* ( ...
		Delta * S2(x).^(2) + 2 * Delta * x .* S2(x) .* S2P(x) + ...
		(1 - Delta) * S2P(x) ...
	);

	Opts = optimset('Display', 'off');

	J1_L = 0;
	J1_R = 0;
	J2_L = 0;
	J2_R = 0;

	Has_LO = false;
	Has_RO = false;

	% 左Outlier
	try
		LO = fzero(ELO, [ 1e-6, X_Min - 1e-6 ], Opts);
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
		RO = fzero(ERO, [ X_Max + 1e-6, 15 ], Opts);
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
	% 用有限上界近似rho -> Inf，避免后续矩阵去噪器出现Inf/Inf。
	% 取1/sqrt(EPS)可让P*、Q*的O(1/rho)量级仍处于可积分范围。
	Rho_Max = 1 / sqrt(EPS);

	X_Min = Para.X_Min;
	X_Max = Para.X_Max;
	Rho = Para.Rho;
	S0 = Para.S0;

	D_U = Para.D_U;
	D_V = Para.D_V;

	Iter_SE = Para.Iter_SE;

	W_U = W_U_Init;
	W_V = W_V_Init;

	for t = 1 : Iter_SE

		MMSE_U = 1 - D_U(W_U);
		MMSE_V = 1 - D_V(W_V);

		if W_U < 1 - EPS && MMSE_U > EPS
			Rho_U = 1 / MMSE_U - 1 / (1 - W_U);
		else
			Rho_U = Rho_Max;
		end

		if W_V < 1 - EPS && MMSE_V > EPS
			Rho_V = 1 / MMSE_V - 1 / (1 - W_V);
		else
			Rho_V = Rho_Max;
		end

		Rho_U = max( 0, min( Rho_Max, real(Rho_U) ) );
		Rho_V = max( 0, min( Rho_Max, real(Rho_V) ) );

		De = @ (x) ( Rho_U * Phi_1(Para, Theta, x) + 1 ) .* ...
			( Rho_V * Phi_2(Para, Theta, x) + Delta ) .* x - ...
			Rho_U * Rho_V * Phi_3(Para, Theta, x).^(2);

		Ps = @ (x) x .* ( Rho_V * Phi_2(Para, Theta, x) + Delta ) ./ De(x);
		Qs = @ (x) Delta * x .* ( Rho_U * Phi_1(Para, Theta, x) + 1 ) ./ De(x);

		E_Ps = integral( @ (x) Ps(x) .* Rho(x), X_Min, X_Max );

		Phi_2_0 = Delta / ( 1 - Theta^(2) * (1 - Delta) * S0 );
		Qs_0 = Delta / (Rho_V * Phi_2_0 + Delta);

		E_Qs_Bulk = integral( @ (x) Qs(x) .* Rho(x), X_Min, X_Max );
		E_Qs = Delta * E_Qs_Bulk + (1 - Delta) * Qs_0;

		Rho_E_Ps = Rho_U * E_Ps;
		Rho_E_Qs = Rho_V * E_Qs;

		if Rho_U <= 0 || ~ isfinite(E_Ps) || E_Ps <= 0 || ...
			~ isfinite(Rho_E_Ps) || Rho_E_Ps <= EPS
			W_U = 0;
		else
			W_U = 1 - (1 - E_Ps) / Rho_E_Ps;
		end

		if Rho_V <= 0 || ~ isfinite(E_Qs) || E_Qs <= 0 || ...
			~ isfinite(Rho_E_Qs) || Rho_E_Qs <= EPS
			W_V = 0;
		else
			W_V = 1 - (1 - E_Qs) / Rho_E_Qs;
		end

		W_U = max( 0, min( 1, W_U ) );
		W_V = max( 0, min( 1, W_V ) );

	end

	M_U = D_U(W_U);
	M_V = D_V(W_V);

end

function [sCos_U, sCos_V, SE_U, SE_V] = OAMP_Iter( ...
	Para, Theta, Has_LO, Has_RO, U_t_Init, V_t_Init, W_U_Init, W_V_Init ...
)

	EPS = Para.EPS;
	Delta = Para.Delta;
	% 与OAMP_SE保持相同的有限rho上界。
	Rho_Max = 1 / sqrt(EPS);

	X_Min = Para.X_Min;
	X_Max = Para.X_Max;
	Rho = Para.Rho;
	S0 = Para.S0;

	D_U = Para.D_U;
	D_V = Para.D_V;

	DMMSE_Est_U = Para.DMMSE_Est_U;
	DMMSE_Est_V = Para.DMMSE_Est_V;
	MMSE_Est_U = Para.MMSE_Est_U;
	MMSE_Est_V = Para.MMSE_Est_V;
	Mean_U = Para.Mean_U;
	Mean_V = Para.Mean_V;

	Iter_OAMP = Para.Iter_OAMP;

	N = Para.N;
	M = Para.M;

	U_0 = Para.U_0;
	V_0 = Para.V_0;

	U_Y = Para.U_Y;
	S_Y = Para.S_Y;
	V_Y = Para.V_Y;

	Singular_Y = Para.Singular_Y;
	Eigen_YYT = Para.Eigen_YYT;

	U_t = U_t_Init;
	V_t = V_t_Init;

	W_U_Array = zeros(1, Iter_OAMP + 1);
	W_V_Array = zeros(1, Iter_OAMP + 1);

	W_U_Array(1) = W_U_Init;
	W_V_Array(1) = W_V_Init;

	for t = 1 : Iter_OAMP

		W_U = W_U_Array(t);
		W_V = W_V_Array(t);

		% DMMSE去噪
		f_U_t = DMMSE_Est_U(U_t, W_U);
		g_V_t = DMMSE_Est_V(V_t, W_V);

		% Oracle归一化: 保持Spec_Rach(7).m的原始实现
		Overlap_U = abs( (f_U_t' * U_0) / M );
		Overlap_V = abs( (g_V_t' * V_0) / N );

		if Overlap_U > EPS
			f_U_t = f_U_t ./ Overlap_U;
		end

		if Overlap_V > EPS
			g_V_t = g_V_t ./ Overlap_V;
		end

		MMSE_U = 1 - D_U(W_U);
		MMSE_V = 1 - D_V(W_V);

		if W_U < 1 - EPS && MMSE_U > EPS
			Rho_U = 1 / MMSE_U - 1 / (1 - W_U);
		else
			Rho_U = Rho_Max;
		end

		if W_V < 1 - EPS && MMSE_V > EPS
			Rho_V = 1 / MMSE_V - 1 / (1 - W_V);
		else
			Rho_V = Rho_Max;
		end

		Rho_U = max( 0, min( Rho_Max, real(Rho_U) ) );
		Rho_V = max( 0, min( Rho_Max, real(Rho_V) ) );

		% 矩阵去噪器P*, Q*, tP*, tQ*
		De = @ (x) ( Rho_U * Phi_1(Para, Theta, x) + 1 ) .* ...
			( Rho_V * Phi_2(Para, Theta, x) + Delta ) .* x - ...
			Rho_U * Rho_V * Phi_3(Para, Theta, x).^(2);

		Ps = @ (x) x .* ( Rho_V * Phi_2(Para, Theta, x) + Delta ) ./ De(x);
		tPs = @ (x) sqrt(Delta) .* Rho_V .* Phi_3(Para, Theta, x) ./ De(x);
		Qs = @ (x) Delta * x .* ( Rho_U * Phi_1(Para, Theta, x) + 1 ) ./ De(x);
		tQs = @ (x) sqrt(Delta) .* Rho_U .* Phi_3(Para, Theta, x) ./ De(x);

		E_Ps = integral( ...
			@ (x) Ps(x) .* Rho(x), X_Min, X_Max, 'ArrayValued', true ...
		);

		Phi_2_0 = Delta / ( 1 - Theta^(2) * (1 - Delta) * S0 );
		Qs_0 = Delta / (Rho_V * Phi_2_0 + Delta);

		E_Qs_Bulk = integral( ...
			@ (x) Qs(x) .* Rho(x), X_Min, X_Max, 'ArrayValued', true ...
		);

		E_Qs = Delta * E_Qs_Bulk + (1 - Delta) * Qs_0;

		Fs_Func = @ (x) 1 - Ps(x) ./ E_Ps;
		tFs_Func = @ (x) tPs(x) ./ E_Ps;
		Gs_Func = @ (x) 1 - Qs(x) ./ E_Qs;
		tGs_Func = @ (x) tQs(x) ./ E_Qs;

		Fs_Val = Fs_Func(Eigen_YYT);
		tFs_Val = tFs_Func(Eigen_YYT);
		Gs_Val_Bulk = Gs_Func(Eigen_YYT);
		tGs_Val_Bulk = tGs_Func(Eigen_YYT);

		% 按Has_RO和Has_LO修正top/bottom Outlier
		if Has_RO
			Fs_Val(1) = 1;
			tFs_Val(1) = 0;
			Gs_Val_Bulk(1) = 1;
			tGs_Val_Bulk(1) = 0;
		end

		if Has_LO
			Fs_Val(end) = 1;
			tFs_Val(end) = 0;
			Gs_Val_Bulk(end) = 1;
			tGs_Val_Bulk(end) = 0;
		end

		Gs_Val_0 = 1 - Qs_0 / E_Qs;
		Gs_Val = [ Gs_Val_Bulk; ones(N - M, 1) * Gs_Val_0 ];
		tGs_Val = [ tGs_Val_Bulk; zeros(N - M, 1) ];

		% 线性更新
		U_Raw = U_Y * ( diag(Fs_Val) * (U_Y' * f_U_t) ) + ...
			U_Y * [ diag(tFs_Val .* Singular_Y), zeros(M, N - M) ] * (V_Y' * g_V_t);

		V_Raw = V_Y * ( diag(Gs_Val) * (V_Y' * g_V_t) ) + ...
			V_Y * diag(tGs_Val) * S_Y' * (U_Y' * f_U_t);

		U_t = sqrt(M) * U_Raw / norm(U_Raw);
		V_t = sqrt(N) * V_Raw / norm(V_Raw);

		% 保持非对称先验所要求的正向标量通道，并同步翻转rank-one信号对
		if abs(Mean_U) > EPS
			Iter_Global_Sign = sign( Mean_U * sum(U_t) );
		elseif abs(Mean_V) > EPS
			Iter_Global_Sign = sign( Mean_V * sum(V_t) );
		else
			Iter_Global_Sign = 1;
		end

		if 0 == Iter_Global_Sign
			Iter_Global_Sign = 1;
		end

		U_t = Iter_Global_Sign * U_t;
		V_t = Iter_Global_Sign * V_t;

		% 伴随经验SE递推
		Rho_E_Ps = Rho_U * E_Ps;
		Rho_E_Qs = Rho_V * E_Qs;

		if Rho_U > 0 && isfinite(E_Ps) && E_Ps > 0 && ...
			isfinite(Rho_E_Ps) && Rho_E_Ps > EPS
			W_U_Array(t + 1) = 1 - (1 - E_Ps) / Rho_E_Ps;
		else
			W_U_Array(t + 1) = 0;
		end

		if Rho_V > 0 && isfinite(E_Qs) && E_Qs > 0 && ...
			isfinite(Rho_E_Qs) && Rho_E_Qs > EPS
			W_V_Array(t + 1) = 1 - (1 - E_Qs) / Rho_E_Qs;
		else
			W_V_Array(t + 1) = 0;
		end

		W_U_Array(t + 1) = max( 0, min( 1, W_U_Array(t + 1) ) );
		W_V_Array(t + 1) = max( 0, min( 1, W_V_Array(t + 1) ) );

	end

	W_U = W_U_Array(end);
	W_V = W_V_Array(end);

	U_Hat = MMSE_Est_U(U_t, W_U);
	V_Hat = MMSE_Est_V(V_t, W_V);

	if norm(U_Hat) > EPS
		Cos_U = (U_Hat' * U_0) / ( norm(U_Hat) * norm(U_0) );
		sCos_U = Cos_U^(2);
	else
		sCos_U = 0;
	end

	if norm(V_Hat) > EPS
		Cos_V = (V_Hat' * V_0) / ( norm(V_Hat) * norm(V_0) );
		sCos_V = Cos_V^(2);
	else
		sCos_V = 0;
	end

	SE_U = D_U(W_U);
	SE_V = D_V(W_V);

end
