unit uNeuralNetworkSequential;

interface

uses
  uSamples, uTypes, uNeuralNetworkBase, System.Classes, System.SysUtils, Mitov.OpenCL;

type
  TNeuralNetworkSequential = class(TNeuralNetworkBase)
  private
    /// <summary>
    ///  M�todo utilizado durante o TEST para registrar no log os valores previstos e calculados da camada de sa�da.
    /// </summary>
    /// <param name="RowSample">
    ///  �ndice da amostra que est� sendo computada.
    /// </param>
    procedure ReportResults(RowSample: Cardinal);
    /// <summary>
    ///  Realiza o c�lculo do somat�rio da multiplica��o dos pesos e seus respectivos neur�nios. Depois utiliza o
    ///  resultado da soma para calcular a fun��o de ativa��o de cada neur�nio. O resultado da computa��o � armazenado
    ///  nos neur�nios de sa�da da camada.
    /// </summary>
    /// <param name="ANeuronsIN">
    ///  Neur�nios de entrada da camada.
    /// </param>
    /// <param name="ANeuronsOUT">
    ///  Neur�nios de sa�da da camada.
    /// </param>
    /// <param name="AWeights">
    ///  Pesos da camada de neur�nios.
    /// </param>
    /// <param name="ASizeIN">
    ///  Quantidade de neur�nios de entrada da camada.
    /// </param>
    /// <param name="ASizeOUT">
    ///  Quantidade de neur�nios de sa�da da camada.
    /// </param>
    procedure ComputeSigmoide(ANeuronsIN, ANeuronsOUT: PVector1D; AWeights: PVector2D; const ASizeIN, ASizeOUT: Word);
  protected
    /// <summary>
    ///  Calcula a etapa de FeedForward do algoritmo de aprendizagem da rede neural. O c�lculo � realizado para todas
    ///  as camadas da rede neural, ou seja, Input -> Hidden e Hidden -> Output.
    /// </summary>
    /// <param name="RowSample">
    ///  �ndice da amostra que est� sendo computada.
    /// </param>
    /// <returns>
    ///  None
    /// </returns>
    /// <remarks>
    ///  Remarks
    /// </remarks>
    procedure FeedForward(RowSample: Cardinal); override;
    /// <summary>
    ///  Calcula a etapa de backpropagation do algoritmo de aprendizagem da rede neural. Nesta etapa, � calculado o
    ///  Delta que representa o quanto a resposta est� diferente do esperado e depois utiliza este valor para atualizar
    ///  os pesos entre os neur�nios, iniciando na camada de sa�da at� a camada de entrada.
    /// </summary>
    /// <param name="RowSample">
    ///  �ndice da amostra que est� sendo computada.
    /// </param>
    procedure BackPropagation(RowSample: Cardinal); override;
  public
    /// <summary>
    ///  M�todo de teste para computar apenas o FeedForward para o conjunto de amostras fornecido no par�metro. Isto �
    ///  �til para verificar se a rede neural consegue prever o valor de sa�da com base no valor de entrada.
    /// </summary>
    /// <param name="ASamplesSet">
    ///  Conjunto de amostras que ser� computado. OBS: as amostras devem ser diferentes do conjunto utilizado para o
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
      // por algum motivo ainda n�o identificado, quando o valor da fun��o acima tende a zero, uma exce��o � lan�ada.
      // para evitar de travar a aplica��o, a exce��o � ignorada e o valor � atribu�do para 0 (zero).
      ANeuronsOUT^[o] := 0;
    end;
  end;
end;

procedure TNeuralNetworkSequential.FeedForward(RowSample: Cardinal);
var
  i: Integer;
begin
  // atribui o valor da amostra nos neur�nios de entrada.  OBS: esta etapa poderia ser otimizada, fazendo com que o
  // c�lculo do m�todo utilizasse diretamente o valor das amostras ao inv�s de copiar para o neur�nio.
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
  // calcula o DELTA da camada de Sa�da
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
