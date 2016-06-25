unit uNeuralNetworkSequential;

interface

uses
  uSamples, uTypes, uNeuralNetworkBase, System.Classes, System.SysUtils, Mitov.OpenCL;

type
  TNeuralNetworkSequential = class(TNeuralNetworkBase)
  private
    /// <summary>
    ///  Método utilizado durante o TEST para registrar no log os valores previstos e calculados da camada de saída.
    /// </summary>
    /// <param name="RowSample">
    ///  Índice da amostra que está sendo computada.
    /// </param>
    procedure ReportResults(RowSample: Cardinal);
    /// <summary>
    ///  Realiza o cálculo do somatório da multiplicação dos pesos e seus respectivos neurônios. Depois utiliza o
    ///  resultado da soma para calcular a função de ativação de cada neurônio. O resultado da computação é armazenado
    ///  nos neurônios de saída da camada.
    /// </summary>
    /// <param name="ANeuronsIN">
    ///  Neurônios de entrada da camada.
    /// </param>
    /// <param name="ANeuronsOUT">
    ///  Neurônios de saída da camada.
    /// </param>
    /// <param name="AWeights">
    ///  Pesos da camada de neurônios.
    /// </param>
    /// <param name="ASizeIN">
    ///  Quantidade de neurônios de entrada da camada.
    /// </param>
    /// <param name="ASizeOUT">
    ///  Quantidade de neurônios de saída da camada.
    /// </param>
    procedure ComputeSigmoide(ANeuronsIN, ANeuronsOUT: PVector1D; AWeights: PVector2D; const ASizeIN, ASizeOUT: Word);
  protected
    /// <summary>
    ///  Calcula a etapa de FeedForward do algoritmo de aprendizagem da rede neural. O cálculo é realizado para todas
    ///  as camadas da rede neural, ou seja, Input -> Hidden e Hidden -> Output.
    /// </summary>
    /// <param name="RowSample">
    ///  Índice da amostra que está sendo computada.
    /// </param>
    /// <returns>
    ///  None
    /// </returns>
    /// <remarks>
    ///  Remarks
    /// </remarks>
    procedure FeedForward(RowSample: Cardinal); override;
    /// <summary>
    ///  Calcula a etapa de backpropagation do algoritmo de aprendizagem da rede neural. Nesta etapa, é calculado o
    ///  Delta que representa o quanto a resposta está diferente do esperado e depois utiliza este valor para atualizar
    ///  os pesos entre os neurônios, iniciando na camada de saída até a camada de entrada.
    /// </summary>
    /// <param name="RowSample">
    ///  Índice da amostra que está sendo computada.
    /// </param>
    procedure BackPropagation(RowSample: Cardinal); override;
  public
    /// <summary>
    ///  Método de teste para computar apenas o FeedForward para o conjunto de amostras fornecido no parâmetro. Isto é
    ///  útil para verificar se a rede neural consegue prever o valor de saída com base no valor de entrada.
    /// </summary>
    /// <param name="ASamplesSet">
    ///  Conjunto de amostras que será computado. OBS: as amostras devem ser diferentes do conjunto utilizado para o
    ///  treinamento da rede neural.
    /// </param>
    procedure Tests;
  end;

implementation

uses
  Math, IdGlobal, Winapi.Windows, uHelpers, System.Threading;

{ TNeuralNetwork }

procedure TNeuralNetworkSequential.ComputeSigmoide(ANeuronsIN, ANeuronsOUT: PVector1D; AWeights: PVector2D;
  const ASizeIN, ASizeOUT: Word);
var
  i, o: Word;
  Sum: Extended;
begin
  for o := 0 to ASizeOUT - 1 do
  begin
    Sum := 0;
    for i := 0 to ASizeIN - 1 do
      Sum := Sum + ANeuronsIN^[i] * AWeights^[i][o];

    try
      ANeuronsOUT^[o] := 1 / (1 + Exp(-Sum));
    except
      // por algum motivo ainda não identificado, quando o valor da função acima tende a zero, uma exceção é lançada.
      // para evitar de travar a aplicação, a exceção é ignorada e o valor é atribuído para 0 (zero).
      ANeuronsOUT^[o] := 0;
    end;
  end;
end;

procedure TNeuralNetworkSequential.FeedForward(RowSample: Cardinal);
var
  i: Integer;
begin
  // atribui o valor da amostra nos neurônios de entrada.  OBS: esta etapa poderia ser otimizada, fazendo com que o
  // cálculo do método utilizasse diretamente o valor das amostras ao invés de copiar para o neurônio.
  for i := 0 to FTopology.Input - 1 do
    FNeuronsInput[i] := FSamplesSet.Samples2D[RowSample][i];

  ComputeSigmoide(@FNeuronsInput,  @FNeuronsHidden, @FWeights2DInputHidden,  FTopology.Input + 1,  FTopology.Hidden);
  ComputeSigmoide(@FNeuronsHidden, @FNeuronsOutput, @FWeights2DHiddenOutput, FTopology.Hidden + 1, FTopology.Output);
end;

procedure TNeuralNetworkSequential.BackPropagation(RowSample: Cardinal);
var
  i, h, o, iOutput: Word;
  Sum: Single;
begin
  // calcula o DELTA da camada de Saída
  for o := 0 to FTopology.Output - 1 do
  begin
    iOutput := FTopology.Input + o;
    FDeltaOutput[o] := FNeuronsOutput[o] * (1 - FNeuronsOutput[o]) * (FSamplesSet.Samples2D[RowSample][iOutput] - FNeuronsOutput[o]);
  end;

  // calcula DELTA da camada Oculta
  for h := 0 to FTopology.Hidden { +1 BIAS } do
  begin
    Sum := 0;
    for o := 0 to FTopology.Output - 1 do
      Sum := Sum + (FDeltaOutput[o] * FWeights2DHiddenOutput[h][o]);

    FDeltaHidden[h] := FNeuronsHidden[h] * (1 - FNeuronsHidden[h]) * Sum;
  end;

  // atualiza Pesos da camada Oculta
  for h := 0 to FTopology.Hidden { +1 BIAS } do
  begin
    for o := 0 to FTopology.Output - 1 do
    begin
      FWeights2DHiddenOutput[h][o] := FWeights2DHiddenOutput[h][o] + FEta * FDeltaOutput[o] * FNeuronsHidden[h];
    end;
  end;

  // atualiza Pesos da camada de Entrada
  for i := 0 to FTopology.Input { +1 BIAS } do
  begin
    for h := 0 to FTopology.Hidden - 1 do
    begin
      FWeights2DInputHidden[i][h] := FWeights2DInputHidden[i][h] + FEta * FDeltaHidden[h] * FNeuronsInput[i];
    end;
  end;
end;

procedure TNeuralNetworkSequential.ReportResults(RowSample: Cardinal);
var
  Info: string;
  i, iOutput: Integer;
begin
  Info := '';
  for i := 0 to FTopology.Input + FTopology.Output - 1 do
    Info := Info + FloatToStr(FSamplesSet.Samples2D[RowSample][i]) + ';';

  for i := 0 to FTopology.Output - 1 do
  begin
    iOutput := FTopology.Input + i;
    Info := Info + ';' + FloatToStr(FNeuronsOutput[i]);
    Info := Info + ';' + FloatToStr(FSamplesSet.Samples2D[RowSample][iOutput] - FNeuronsOutput[i]);
  end;
  FLog.Add(Info);
end;

procedure TNeuralNetworkSequential.Tests;
var
  Row: Integer;
begin
  for Row := 0 to FSamplesSet.SamplesCount - 1 do
  begin
    FeedForward(Row);
    ReportResults(Row);
  end;
end;

end.
