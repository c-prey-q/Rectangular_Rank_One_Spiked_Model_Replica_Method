clear;
clc;
close all;
warning('off', 'MATLAB:integral:NonFiniteValue');
warning('off', 'MATLAB:fzero:NAx');

format long;

load('Saddle_Point_Equation_Rad-1-3_2Points-0.083333_Gauss_Ber-0.4_A-2_T-0.2-0.001-0.34_I-1000.mat');

EPS = 1e-12;

Global_Optimal = []; % 存储所有 t 下的最高 FE 行 (不含 FE 列)

Sub_Optimal = []; % 存储所有 t 下被剔除掉的次优解 (不含 FE 列)

for t = 1 : Num_Theta

	Theta = Theta_Array(t);

	Solution = Solution_Array{t};

	Solution(:, end + 1) = Theta * ones( size(Solution, 1), 1 );
	if isempty(Solution)
		continue;
	end

	Valid_Rows = Solution( Solution(:, 14) < EPS, : );
	if isempty(Valid_Rows)
		continue;
	end

	[Sorted_W, Sort_Idx] = sort( Valid_Rows(:, 5) );
	Sorted_Rows = Valid_Rows(Sort_Idx, :);

	Diff_W = diff(Sorted_W);
	Keep_Mask = [true; Diff_W > EPS];
	Unique_Rows = Sorted_Rows(Keep_Mask, :);

	[Max_Fe_Val, Max_Fe_Idx] = max( Unique_Rows(:, 9) );

	Best_Row = Unique_Rows(Max_Fe_Idx, :);

	Others_Mask = true( size(Unique_Rows, 1), 1 );
	Others_Mask(Max_Fe_Idx) = false;
	Other_Rows = Unique_Rows(Others_Mask, :);

	Global_Optimal = [Global_Optimal; Best_Row];
	Sub_Optimal = [Sub_Optimal; Other_Rows];

end

Global_Optimal = [ Global_Optimal(:, end), Global_Optimal(:, 1 : end - 1) ];
Sub_Optimal = [ Sub_Optimal(:, end), Sub_Optimal(:, 1 : end - 1) ];