function J = computarCusto(X, y, theta)
%COMPUTARCUSTO Calcula o custo da regressao linear
%   J = COMPUTARCUSTO(X, y, theta) calcula o custo de usar theta como 
%   parametro da regressao linear para ajustar os dados de X e y

% Initializa algumas variaveis uteis
m = length(y); % numero de exemplos de treinamento

% Voce precisa retornar a seguinte variavel corretamente
J = 0;

% ====================== ESCREVA O SEU CODIGO AQUI ======================
% Instrucoes: Calcule o custo de uma escolha particular de theta.
%             Voce precisa armazenar o valor do custo em J.

%hyp_func(x) = theta(1) + (theta(2)*x);

hyp_func = zeros(m,1);
hyp_func(1:m) = (theta(1) + (theta(2) * X(1:m, [2])));

hyp_func = hyp_func - y;
hyp_func = hyp_func .* hyp_func;
hyp_func = sum(hyp_func) / (2*m);

J = hyp_func;
% =========================================================================

end
