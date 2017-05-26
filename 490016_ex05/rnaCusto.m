function [J grad] = rnaCusto(nn_params, ...
                             input_layer_size, ...
                             hidden_layer_size, ...
                             num_labels, ...
                             X, y, lambda)
%RNACUSTO Implementa a funcao de custo para a rede neural com duas camadas
%voltada para tarefa de classificacao
%   [J grad] = RNACUSTO(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) calcula o custo e gradiente da rede neural. The
%   Os parametros da rede neural sao colocados no vetor nn_params
%   e precisam ser transformados de volta nas matrizes de peso.
%
%   input_layer_size - tamanho da camada de entrada
%   hidden_layer_size - tamanho da camada oculta
%   num_labels - numero de classes possiveis
%   lambda - parametro de regularizacao
%
%   O vetor grad de retorno contem todas as derivadas parciais
%   da rede neural.
%

% Extrai os parametros de nn_params e alimenta as variaveis Theta1 e Theta2.
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Definindo variaveis uteis
m = size(X, 1)

% As variaveis a seguir precisam ser retornadas corretamente
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== INSIRA SEU CODIGO AQUI ======================
% Instrucoes: Voce deve completar o codigo a partir daqui 
%               acompanhando os seguintes passos.
%
% (1): Lembre-se de transformar os rotulos Y em vetores com 10 posicoes,
%      onde tera zero em todas posicoes exceto na posicao do rotulo
%
% (2): Execute a etapa de feedforward e coloque o custo na variavel J.
%      Apos terminar, verifique se sua funcao de custo esta correta,
%      comparando com o custo calculado em ex05.m.
%
% (3): Implemente o algoritmo de backpropagation para calcular 
%      os gradientes e alimentar as variaveis Theta1_grad e Theta2_grad.
%      Ao terminar essa etapa, voce pode verificar se sua implementacao 
%      esta correta atraves usando a funcao verificaGradiente.
%
% (4): Implemente a regularização na função de custo e gradiente.
%
%size(nn_params)
%theta1
%input_layer_size
%theta2
%hidden_layer_size
% X == 5000 x 400
% X' == 400 x 5000
% Theta1 == 25 x 401
% Theta2 == 10 x 26
% Y == 5000 x 10


Y = zeros(size(y), 10);
size(y)
for i = 1:10
 Y(:,i) = (y==i);
endfor

size(Y)


hyp = X * Theta1(:,2:401)';
J_theta += (-(Y') * log(hyp) - (1 - Y)' * log(1-hyp));

J_theta1 = max(J_theta' * ones(10,1));

hyp = hyp * Theta2(:,2:26)';
J_theta2 = (-(Y') * log(hyp) - (1 - Y)' * log(1-hyp));
J_theta2 = max(J_theta2' * ones(10,1));

J = J_theta1 + J_theta2;

% -------------------------------------------------------------



% =========================================================================

% Junta os gradientes
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
