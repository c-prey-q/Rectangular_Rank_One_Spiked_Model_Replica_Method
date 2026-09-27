% Q_Num = 120;
% Q_Contour = 256;

% Para.Q_Num = Q_Num;
% Para.Q_Contour = Q_Contour;

function Res = Compute_Free_Energy(Para, Lambda, w_u, w_v, m_u, m_v)
% Compute_Free_Energy
% Numerical evaluation of the replica free entropy after solving the
% self-consistent auxiliary functions \tilde{M}.
%
% Important conventions:
% 1. The positive spectral variable is y = s^(2).
% 2. The measure is
% tilde_mu = (1 / Alpha) mu + (1 - 1 / Alpha) delta_0.
% 3. The unknown vector X contains only the positive spectral nodes and one
% zero-atom unknown X_vv_0 = \tilde{M}_{vv}(0).
% 4. x^(2) always means complex square x * x, not abs(x)^(2).

	EPS = Para.EPS;

	Alpha = Para.Alpha;

	Rho = Para.Rho;
	H = Para.H;
	S0 = Para.S0;

	w_u = max( EPS, min(1 - EPS, w_u) );
	w_v = max( EPS, min(1 - EPS, w_v) );

	hm_u = w_u / max(EPS, 1 - w_u);
	hm_v = w_v / max(EPS, 1 - w_v);

	hm_u = max(EPS, hm_u);
	hm_v = max(EPS, hm_v);

	mmse_u = 1 - m_u;
	mmse_v = 1 - m_v;

	mmse_u = max(EPS, mmse_u);
	mmse_v = max(EPS, mmse_v);

	rho_u = 1 / max(EPS, mmse_u) - 1 / max(EPS, 1 - w_u);
	rho_u = max(EPS, rho_u);

	rho_v = 1 / max(EPS, mmse_v) - 1 / max(EPS, 1 - w_v);
	rho_v = max(EPS, rho_v);

	tm_u = - rho_u;
	tm_v = - rho_v;

	FP.w_u = w_u;
	FP.w_v = w_v;

	FP.m_u = m_u;
	FP.m_v = m_v;

	FP.tm_u = tm_u;
	FP.tm_v = tm_v;

	J_UU = @ (x) Lambda^(2) * pi^(2) * x .* ( ( Rho(x) ).^(2) + ( H(x) ).^(2) ) + ...
		(Alpha - 1) * Lambda^(2) * 2 * pi * H(x) + ...
		(Alpha - 1)^(2) * Lambda^(2) * 1 ./ x;

	J_VV = @ (x) Lambda^(2) * pi^(2) * x .* ( ( Rho(x) ).^(2) + ( H(x) ).^(2) );

	a_V = - (Alpha - 1) * Lambda^(2) * S0;

	TG_U = @(s) 1 - tm_u + J_UU(s.^(2));
	TG_V = @(s) 1 - tm_v + J_VV(s.^(2));

	TG_V0 = 1 - tm_v + a_V;

	TG_UV = @(s) - 1 / 2 * ( ...
		Lambda * 2 * pi * s .* H(s.^(2)) + (Alpha - 1) * Lambda ./ s ...
	);

	FP.TG_U = TG_U;
	FP.TG_V = TG_V;
	FP.TG_V0 = TG_V0;
	FP.TG_UV = TG_UV;

	Grid = Build_Grid(Para);

	[x_I, Info] = Solve_tM(Para, Lambda, FP, Grid);

	Cond = Info.Cond;
	Residual = Info.Residual;

	F_Val = Evaluate_Free_Entropy(Para, Lambda, FP, Grid, x_I);

	fprintf( 'cond(I - K): %.3e\n', Cond );
	fprintf( 'Self-consistency Residual: %.3e\n', Residual );
	fprintf( 'Free entropy real(f): %.15g\n', real(F_Val) );
	fprintf( 'Free entropy imag(f): %.3e\n', imag(F_Val) );

	Res = real(F_Val);

end

function Grid = Build_Grid(Para)

	Alpha = Para.Alpha;

	X_Min = Para.X_Min;
	X_Max = Para.X_Max;
	Rho = Para.Rho;

	Q_Num = Para.Q_Num;
	Q_Contour = Para.Q_Contour;

	[Y_Lt, Dy_Lt] = Gauss_Legendre_Rule(X_Min, X_Max, Q_Num);
	W_Lt = Dy_Lt .* Rho(Y_Lt);
	S_Lt = sqrt(Y_Lt);

	Zero_Mass = 1 - 1 / Alpha;

	tW_Bulk_Lt = 1 / Alpha * W_Lt;

	Num_S = length(S_Lt);

	Grid.W_Lt = W_Lt;
	Grid.S_Lt = S_Lt;

	Grid.Zero_Mass = Zero_Mass;

	Grid.tW_Bulk_Lt = tW_Bulk_Lt;

	Grid.Num_S = Num_S;

	%%%%%%

	Center = (X_Min + X_Max) / 2;
	Radius = (X_Max - X_Min) / 2;
	R = (1 + 0.1) * Radius;

	if Center - R <= 0
		error('The chosen y-contour includes 0. Reduce Margin or choose a > 0 with enough gap.');
	end

	Te_Lt = (0 : Q_Contour - 1).' * (2 * pi / Q_Contour);
	Dte = 2 * pi / Q_Contour;

	Yte_Lt = Center + R * exp(1i * Te_Lt);
	Dyte_Lt = 1i * R * exp(1i * Te_Lt);

	Xte_P_Lt = sqrt(Yte_Lt);
	Xte_M_Lt = - Xte_P_Lt;

	Cw_P_Lt = 1 / (2 * pi * 1i) * Dte * Dyte_Lt ./ (2 * Xte_P_Lt);
	Cw_M_Lt = 1 / (2 * pi * 1i) * Dte * Dyte_Lt ./ (2 * Xte_M_Lt);

	Xte_Lt = [Xte_P_Lt; Xte_M_Lt];
	Cw_Lt = [Cw_P_Lt; Cw_M_Lt];

	Grid.Xte_Lt = Xte_Lt;
	Grid.Cw_Lt = Cw_Lt;

end

function [x_I, Info] = Solve_tM(Para, Lambda, FP, Grid)

	Num_S = Grid.Num_S;
	Num = 4 * Num_S + 1;

	Tmp = zeros(Num, 1);

	g = Compute_F(Para, Lambda, FP, Grid, Tmp);

	K = zeros(Num, Num);

	for j = 1 : Num

		Tmp = zeros(Num, 1);
		Tmp(j) = 1;

		Kj = Compute_F(Para, Lambda, FP, Grid, Tmp) - g;
		K(:, j) = Kj;

	end

	A = eye(Num) - K;

	Cond = cond(A);

	x_I = A \ g;

	x_O = Compute_F(Para, Lambda, FP, Grid, x_I);
	Residual = norm(x_I - x_O) / ( 1 + norm(x_I) );

	Info.Cond = Cond;
	Info.Residual = Residual;

end

function x_O = Compute_F(Para, Lambda, FP, Grid, x_I)

	[Pack, ~] = Forward_Pass(Para, Lambda, FP, Grid, x_I);

	x_uu = Pack.x_uu;
	x_uv = Pack.x_uv;
	x_vu = Pack.x_vu;
	x_vv = Pack.x_vv;

	x_vv_0 = Pack.x_vv_0;

	x_O = Pack_X(x_uu, x_uv, x_vu, x_vv, x_vv_0);

end

function [Pack, Data] = Forward_Pass(Para, Lambda, FP, Grid, x_I)

	Alpha = Para.Alpha;

	VP = Para.VP;

	tm_u = FP.tm_u;
	tm_v = FP.tm_v;

	TG_U = FP.TG_U;
	TG_V = FP.TG_V;
	TG_V0 = FP.TG_V0;
	TG_UV = FP.TG_UV;

	W_Lt = Grid.W_Lt;
	S_Lt = Grid.S_Lt;

	Zero_Mass = Grid.Zero_Mass;

	tW_Bulk_Lt = Grid.tW_Bulk_Lt;

	Num_S = Grid.Num_S;

	%%%%%%

	Xte_Lt = Grid.Xte_Lt;
	Cw_Lt = Grid.Cw_Lt;

	Pack = Unpack_X(x_I, Num_S);

	x_uu_I = Pack.x_uu;
	x_uv_I = Pack.x_uv;
	x_vu_I = Pack.x_vu;
	x_vv_I = Pack.x_vv;

	x_vv_0_I = Pack.x_vv_0;

	tg_u_Lt = TG_U(S_Lt);
	tg_v_Lt = TG_V(S_Lt);
	tg_uv_Lt = TG_UV(S_Lt);

	oD_Lt = tg_v_Lt - 4 * tg_uv_Lt.^(2) ./ tg_u_Lt;

	tm_u_Lt = tm_u + x_uu_I;
	tm_v_Lt = tm_v + x_vv_I;

	tm_v_0 = tm_v + x_vv_0_I;

	a_uu_Lt = - 1 ./ tg_u_Lt .* ( ...
		tm_u_Lt - 2 * tg_uv_Lt ./ oD_Lt .* ( ...
			x_vu_I - 2 * tm_u_Lt .* tg_uv_Lt ./ tg_u_Lt ...
		) ...
	);

	a_uv_Lt = - 1 ./ tg_u_Lt .* ( ...
		x_uv_I - 2 * tg_uv_Lt ./ oD_Lt .* ( ...
			tm_v_Lt - 2 * x_uv_I .* tg_uv_Lt ./ tg_u_Lt ...
		) ...
	);

	a_vu_Lt = - 1 ./ oD_Lt .* ( ...
		x_vu_I - 2 * tm_u_Lt .* tg_uv_Lt ./ tg_u_Lt ...
	);

	a_vv_Lt = - Alpha ./ oD_Lt .* ( ...
		tm_v_Lt - 2 * x_uv_I .* tg_uv_Lt ./ tg_u_Lt ...
	);

	a_vv_0 = - Alpha / TG_V0 * tm_v_0;

	% Q_Num = 120;
    % Q_Contour = 256;
    % 2 * Q_Contour x Q_Num
	Den_Lt = Xte_Lt.^(2) - (S_Lt.').^(2);
	Xte_Den_Lt = Xte_Lt ./ Den_Lt;
	S_Den_Lt = S_Lt.' ./ Den_Lt;

	M_uu_Lt = Xte_Den_Lt * (W_Lt .* a_uu_Lt);
	M_uv_Lt = S_Den_Lt * (W_Lt .* a_uv_Lt);
	M_vu_Lt = S_Den_Lt * (W_Lt .* a_vu_Lt);
	M_vv_Lt = Xte_Den_Lt * (tW_Bulk_Lt .* a_vv_Lt) ...
		+ 1 ./ Xte_Lt * (Zero_Mass * a_vv_0);

	c1_Lt = Xte_Den_Lt * W_Lt;
	c2_Lt = Alpha * ( ...
		Xte_Den_Lt * tW_Bulk_Lt + (1 ./ Xte_Lt) * Zero_Mass ...
	);

	D_Lt = c1_Lt .* c2_Lt - 1 / Lambda^(2);

	VP_Xte_Lt = VP(Xte_Lt.^(2));
	Fc_Lt = - 2 * Lambda^(2) * Xte_Lt .* VP_Xte_Lt ./ D_Lt;

	T_uu_Lt = c2_Lt.^(2) .* M_uu_Lt - 1 / Lambda * c2_Lt .* M_vu_Lt ...
		+ 1 / Lambda * c2_Lt .* M_uv_Lt - 1 / Lambda^(2) * M_vv_Lt;

	T_uv_Lt = 1 / Lambda * c2_Lt .* M_uu_Lt - 1 / Lambda^(2) * M_vu_Lt ...
		+ c1_Lt .* c2_Lt .* M_uv_Lt - 1 / Lambda * c1_Lt .* M_vv_Lt;

	T_vu_Lt = - 1 / Lambda * c2_Lt .* M_uu_Lt + c1_Lt .* c2_Lt .* M_vu_Lt ...
		- 1 / Lambda^(2) * M_uv_Lt + 1 / Lambda * c1_Lt .* M_vv_Lt;

	T_vv_Lt = - 1 / Lambda^(2) * M_uu_Lt + 1 / Lambda * c1_Lt .* M_vu_Lt ...
		- 1 / Lambda * c1_Lt .* M_uv_Lt + c1_Lt.^(2) .* M_vv_Lt;

	hM_uu_Lt = Fc_Lt .* T_uu_Lt;
	hM_uv_Lt = Fc_Lt .* T_uv_Lt;
	hM_vu_Lt = Fc_Lt .* T_vu_Lt;
	hM_vv_Lt = Fc_Lt .* T_vv_Lt;

	Rec_Lt = 1 ./ Den_Lt;

	x_uu_O = ( ( Cw_Lt .* hM_uu_Lt .* Xte_Lt ).' * Rec_Lt ).';
	x_uv_O = ( ( Cw_Lt .* hM_uv_Lt ).' * ( Rec_Lt .* S_Lt.' ) ).';
	x_vu_O = ( ( Cw_Lt .* hM_vu_Lt ).' * ( Rec_Lt .* S_Lt.' ) ).';
	x_vv_O = ( ( Cw_Lt .* hM_vv_Lt .* Xte_Lt ).' * Rec_Lt ).';

	x_vv_0_O = sum( Cw_Lt .* hM_vv_Lt ./ Xte_Lt );

	Pack.x_uu = x_uu_O;
	Pack.x_uv = x_uv_O;
	Pack.x_vu = x_vu_O;
	Pack.x_vv = x_vv_O;

	Pack.x_vv_0 = x_vv_0_O;

	Data.tg_u_Lt = tg_u_Lt;
	Data.tg_v_Lt = tg_v_Lt;
	Data.tg_uv_Lt = tg_uv_Lt;

	Data.oD_Lt = oD_Lt;

	Data.M_uu_Lt = M_uu_Lt;
	Data.M_uv_Lt = M_uv_Lt;
	Data.M_vu_Lt = M_vu_Lt;
	Data.M_vv_Lt = M_vv_Lt;

	Data.hM_uu_Lt = hM_uu_Lt;
	Data.hM_uv_Lt = hM_uv_Lt;
	Data.hM_vu_Lt = hM_vu_Lt;
	Data.hM_vv_Lt = hM_vv_Lt;

end

function F_Val = Evaluate_Free_Entropy(Para, Lambda, FP, Grid, x_I)

	Alpha = Para.Alpha;

	E_Log_P_U = Para.E_Log_P_U;
	E_Log_P_V = Para.E_Log_P_V;

	w_u = FP.w_u;
	w_v = FP.w_v;

	m_u = FP.m_u;
	m_v = FP.m_v;

	tm_u = FP.tm_u;
	tm_v = FP.tm_v;

	TG_V0 = FP.TG_V0;

	W_Lt = Grid.W_Lt;

	Zero_Mass = Grid.Zero_Mass;

	Num_S = Grid.Num_S;

	Cw_Lt = Grid.Cw_Lt;

	[~, Data] = Forward_Pass(Para, Lambda, FP, Grid, x_I);

	tg_u_Lt = Data.tg_u_Lt;
	tg_uv_Lt = Data.tg_uv_Lt;

	oD_Lt = Data.oD_Lt;

	M_uu_Lt = Data.M_uu_Lt;
	M_uv_Lt = Data.M_uv_Lt;
	M_vu_Lt = Data.M_vu_Lt;
	M_vv_Lt = Data.M_vv_Lt;

	hM_uu_Lt = Data.hM_uu_Lt;
	hM_uv_Lt = Data.hM_uv_Lt;
	hM_vu_Lt = Data.hM_vu_Lt;
	hM_vv_Lt = Data.hM_vv_Lt;

	Pack = Unpack_X(x_I, Num_S);

	x_uu = Pack.x_uu;
	x_uv = Pack.x_uv;
	x_vu = Pack.x_vu;
	x_vv = Pack.x_vv;

	x_vv_0 = Pack.x_vv_0;

	tm_u_Lt = tm_u + x_uu;
	tm_v_Lt = tm_v + x_vv;

	tm_v_0 = tm_v + x_vv_0;

	Trace_Term = hM_uu_Lt .* M_uu_Lt ...
		+ hM_uv_Lt .* M_uv_Lt ...
		+ hM_vu_Lt .* M_vu_Lt ...
		+ hM_vv_Lt .* M_vv_Lt;

	I_M = 1 / 2 * sum(Cw_Lt .* Trace_Term);

	I_0 = - 1 / 2 * Alpha * Zero_Mass * ( ...
		log(TG_V0) + ( tm_v - tm_v_0^(2) ) / TG_V0 ...
	);

	oN_Lt = tm_v - x_vu.^(2) + 2 * tg_uv_Lt .* ( ...
		2 * tm_u_Lt .* x_vu ./ tg_u_Lt ...
		+ 2 * ( tm_u - tm_u_Lt.^(2) ) .* tg_uv_Lt ./ tg_u_Lt.^(2) ...
	) - ( ...
		tm_v_Lt - 2 * x_uv .* tg_uv_Lt ./ tg_u_Lt ...
	).^(2);

	Bulk_Int = log(tg_u_Lt) + ( tm_u - tm_u_Lt.^(2) ) ./ tg_u_Lt ...
		+ oN_Lt ./ oD_Lt - x_uv.^(2) ./ tg_u_Lt + log(oD_Lt);

	I_Mu = - 1 / 2 * sum(W_Lt .* Bulk_Int);

	I_u = E_Log_P_U(w_u);
	I_v = E_Log_P_V(w_v);

	Scalar_Part = - 1 / 2 * log(1 - m_u) ...
		- Alpha / 2 * log(1 - m_v) ...
		- 1 / ( 2 * (1 - m_u) ) ...
		- Alpha / ( 2 * (1 - m_v) ) ...
		+ I_u + Alpha * I_v;

	F_Val = I_M + Scalar_Part + I_0 + I_Mu;

end

function x_O = Pack_X(x_uu, x_uv, x_vu, x_vv, x_vv_0)

	Num_S = length(x_uu);

	x_O = zeros(4 * Num_S + 1, 1);

	x_O(1 : 4 : 4 * Num_S) = x_uu;
	x_O(2 : 4 : 4 * Num_S) = x_uv;
	x_O(3 : 4 : 4 * Num_S) = x_vu;
	x_O(4 : 4 : 4 * Num_S) = x_vv;

	x_O(4 * Num_S + 1) = x_vv_0;

end

function Pack = Unpack_X(x_I, Num_S)

	x_uu = x_I(1 : 4 : 4 * Num_S);
	x_uv = x_I(2 : 4 : 4 * Num_S);
	x_vu = x_I(3 : 4 : 4 * Num_S);
	x_vv = x_I(4 : 4 : 4 * Num_S);

	x_vv_0 = x_I(4 * Num_S + 1);

	Pack.x_uu = x_uu;
	Pack.x_uv = x_uv;
	Pack.x_vu = x_vu;
	Pack.x_vv = x_vv;

	Pack.x_vv_0 = x_vv_0;

end

function [Y_Lt, Dy_Lt] = Gauss_Legendre_Rule(X_Min, X_Max, Q_Num)

	Beta = (1 : Q_Num - 1).' ./ sqrt( ...
		4 * (1 : Q_Num - 1)'.^(2) - 1 ...
	);

	J = diag(Beta, 1) + diag(Beta, - 1);

	[V, D] = eig(J);

	x0 = diag(D);
	[x0, Idx] = sort(x0);
	V = V(:, Idx);

	w0 = 2 * ( V(1, :)' ).^(2);

	Y_Lt = (X_Max - X_Min) / 2 * x0 + (X_Min + X_Max) / 2;
	Dy_Lt = (X_Max - X_Min) / 2 * w0;

end
