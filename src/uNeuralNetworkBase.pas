unit uNeuralNetworkBase;

interface

uses
  System.Classes, uSamples, uTypes;

const
  ETA_DEFAULT = 0.15;

type
  TTopology = record
    Input: Word;
    Hidden: Word;
    Output: Word;
  end;

  TNeuralNetworkBase = class
  protected
    // dados e configura��es da rede neural
    FEta: Single;
    FTopology: TTopology;
    FSamplesSet: TSamplesSet;

    // armazena o valor de sa�da de cada neur�nio
    FNeuronsInput: TVector1D;
    FNeuronsHidden: TVector1D;
    FNeuronsOutput: TVector1D;

    // pesos entre os neur�nios
    FWeights2DInputHidden: TVector2D;
    FWeights2DHiddenOutput: TVector2D;
    // pesos entre os neur�nios "vetorizado"
    FWeights1DInputHidden: TVector1D;
    FWeights1DHiddenOutput: TVector1D;

    // valores Delta (diferen�a entre computado e esperado) das camadas
    FDeltaHidden: TVector1D;
    FDeltaOutput: TVector1D;

    FLog: TStrings;

    /// <summary>
    ///  Gera um valor aleat�rio entre -1 e +1.
    /// </summary>
    /// <returns>
    ///  Single
    /// </returns>
    function GetRandomWeight: Single;
    procedure FeedForward(iSample: Cardinal); virtual; abstract;
    procedure BackPropagation(iSample: Cardinal); virtual; abstract;
  public
    constructor Create(ATopology: TTopology); virtual;
    destructor Destroy; override;

    /// <summary>
    ///  Define pesos aleat�rios para cada conex�o entre os neur�nios.
    /// </summary>
    procedure DefineRandomWeights;
    /// <summary>
    ///  Salva os pesos das conex�o entre os neur�nios em arquivo, no formato CSV.
    /// </summary>
    /// <param name="AFileName">
    ///  Caminho completo do arquivo que ser� criado para salvar o valor dos pesos.
    /// </param>
    procedure SaveWeights(const AFileName: string);
    /// <summary>
    ///  Carrega, a partir de um arquivo no formato CSV, o peso das conex�es entre os neur�nios.
    /// </summary>
    /// <param name="AFileName">
    ///  Caminho completo do arquivo no formato CSV.
    /// </param>
    /// <remarks>
    ///  Os dados devem corresponder a mesma topologia da rede neural.
    /// </remarks>
    procedure LoadWeights(const AFileName: string);
    /// <summary>
    ///  Executa 1 �poca da etapa de aprendizagem, ou seja, computa o FeedForward e Backpropagation para todas as
    ///  entradas do conjunto de amostras.
    /// </summary>
    /// <remarks>
    ///  A condi��o de parada do m�todo � computar todas as entradas do conjunto de amostras. N�o h� um controle de
    ///  parada com base na margem de erro do previsto e computado.
    /// </remarks>
    procedure Learn;

    /// <summary>
    ///  ETA representa o coeficiente de aprendizagem da rede, que varia de 0 a 1. O valor padr�o � 0,15.
    /// </summary>
    property Eta: Single read FEta write FEta;
    property Log: TStrings read FLog write FLog;
    property SamplesSet: TSamplesSet read FSamplesSet write FSamplesSet;
  end;

implementation

uses
  System.Math, System.SysUtils, uHelpers, IdGlobal;

{ TNeuralNetworkBase }

constructor TNeuralNetworkBase.Create(ATopology: TTopology);
begin
  FEta := ETA_DEFAULT;
  FTopology := ATopology;
  FSamplesSet := nil;
  FLog := nil;

  SetLength(FNeuronsInput, FTopology.Input + 1); // +1 for BIAS
  SetLength(FNeuronsHidden, FTopology.Hidden + 1); // +1 for BIAS
  SetLength(FNeuronsOutput, FTopology.Output);

  SetLength(FWeights2DInputHidden, FTopology.Input + 1, FTopology.Hidden);
  SetLength(FWeights2DHiddenOutput, FTopology.Hidden + 1, FTopology.Output);

  SetLength(FWeights1DInputHidden, (FTopology.Input + 1) * FTopology.Hidden);
  SetLength(FWeights1DHiddenOutput, (FTopology.Hidden + 1) * FTopology.Output);

  SetLength(FDeltaHidden, FTopology.Hidden + 1); // +1 for BIAS
  SetLength(FDeltaOutput, FTopology.Output);

  // BIAS fixed value
  FNeuronsInput[FTopology.Input] := 1;
  FNeuronsHidden[FTopology.Hidden] := 1;
end;

destructor TNeuralNetworkBase.Destroy;
begin
  SetLength(FNeuronsInput, 0);
  SetLength(FNeuronsHidden, 0);
  SetLength(FNeuronsOutput, 0);

  SetLength(FWeights2DInputHidden, 0, 0);
  SetLength(FWeights2DHiddenOutput, 0, 0);

  SetLength(FWeights1DInputHidden, 0);
  SetLength(FWeights1DHiddenOutput, 0);

  SetLength(FDeltaHidden, 0);
  SetLength(FDeltaOutput, 0);

  inherited;
end;

function TNeuralNetworkBase.GetRandomWeight: Single;
begin
  // the weights needs to be initialize with values in range -1 to +1
  Result := RandomRange(-1000, 1000) / 1000;
end;

procedure TNeuralNetworkBase.DefineRandomWeights;
var
  i, h, o, k: Word;
  Filename: string;
begin
  //FLog.Add('DefineRandomWeights');
  Randomize;

  for i := 0 to FTopology.Input {+1 BIAS} do
  begin
    for h := 0 to FTopology.Hidden - 1 do
    begin
      FWeights2DInputHidden[i][h] := GetRandomWeight;
      //FLog.AddFmt('FWeights2DInputHidden[%d][%d]=%.6f', [i, h, FWeights2DInputHidden[i][h]]);

      k := i * FTopology.Hidden + h;
      FWeights1DInputHidden[k] := FWeights2DInputHidden[i][h];
      //FLog.AddFmt('FWeights1DInputHidden[%d]=%.6f', [k, FWeights1DInputHidden[k]]);
    end;
  end;

  for h := 0 to FTopology.Hidden {+1 BIAS} do
  begin
    for o := 0 to FTopology.Output - 1 do
    begin
      FWeights2DHiddenOutput[h][o] := GetRandomWeight;
      //FLog.AddFmt('FWeights2DHiddenOutput[%d][%d]=%.6f', [h, o, FWeights2DHiddenOutput[h][o]]);

      k := h * FTopology.Output + o;
      FWeights1DHiddenOutput[k] := FWeights2DHiddenOutput[h][o];
      //FLog.AddFmt('FWeights1DHiddenOutput[%d]=%.6f', [k, FWeights1DHiddenOutput[k]]);
    end;
  end;
end;

procedure TNeuralNetworkBase.SaveWeights(const AFileName: string);
var
  Weights: TStringList;
  Line: string;
  i, h, o: Integer;
begin
  try
    Weights := TStringList.Create;

    for i := 0 to FTopology.Input {+ BIAS} do
    begin
      Line := FloatToStr(FWeights2DInputHidden[i][0]);
      for h := 1 to FTopology.Hidden - 1 do
        Line := Line + ';' + FloatToStr(FWeights2DInputHidden[i][h]);

      Weights.Add(Line);
    end;

    for h := 0 to FTopology.Hidden {+ BIAS} do
    begin
      Line := FloatToStr(FWeights2DHiddenOutput[h][0]);
      for o := 1 to FTopology.Output - 1 do
        Line := Line + ';' + FloatToStr(FWeights2DHiddenOutput[h][o]);

      Weights.Add(Line);
    end;

    Weights.SaveToFile(AFileName);
  finally
    FreeAndNil(Weights);
  end;
end;

procedure TNeuralNetworkBase.LoadWeights(const AFileName: string);
var
  Weights: TStringList;
  i, h, o, k: Integer;
  Line: string;
  Value: string;
begin
  Weights := TStringList.Create;
  try
    Weights.LoadFromFile(AFileName);

    for i := 0 to FTopology.Input {+ BIAS} do
    begin
      Line := Weights.Strings[0];
      for h := 0 to FTopology.Hidden - 1 do
      begin
        Value := Fetch(Line, ';');
        FWeights2DInputHidden[i][h] := StrToFloat(Value);

        k := i * FTopology.Hidden + h;
        FWeights1DInputHidden[k] := FWeights2DInputHidden[i][h];
      end;
      Weights.Delete(0);
    end;

    for h := 0 to FTopology.Hidden {+ BIAS} do
    begin
      Line := Weights.Strings[0];
      for o := 0 to FTopology.Output - 1 do
      begin
        Value := Fetch(Line, ';');
        FWeights2DHiddenOutput[h][o] := StrToFloat(Value);

        k := h * FTopology.Output + o;
        FWeights1DHiddenOutput[k] := FWeights2DHiddenOutput[h][o];
      end;
      Weights.Delete(0);
    end;
  finally
    FreeAndNil(Weights);
  end;
end;

procedure TNeuralNetworkBase.Learn;
var
  Row: Integer;
  Sample: PVector1D;
  Error: Single;
begin
  Error := 0;
  for Row := 0 to FSamplesSet.SamplesCount - 1 do
  begin
    FeedForward(Row);
    BackPropagation(Row);

    Sample := @FSamplesSet.Samples2D[Row];
    //Error := Error + Abs((Sample^[FTopology.Input]) - Abs(FNeuronsOutput[0]));
    Error := Error + Power(Sample^[FTopology.Input] - FNeuronsOutput[0], 2);
  end;
  //Error := Error / FSamplesSet.SamplesCount;
  Error := Error / 2;
  FLog.Add(FloatToStr(Error));
end;

end.
