unit uNeuralNetworkBase;

interface

uses
  System.Classes, uSamples;

type
  TTopology = record
    Input: Word;
    Hidden: Word;
    Output: Word;
  end;

  TNeuralNetworkBase = class
  protected
    FTopology: TTopology;

    // armazena o valor de saída de cada neurônio
    FNeuronsInput: array of Single;
    FNeuronsHidden: array of Single;
    FNeuronsOutput: array of Single;

    // pesos entre os neurônios
    FWeightsInputHidden: array of array of Single;
    FWeightsHiddenOutput: array of array of Single;
    // pesos entre os neurônios vetorizado
    FWInputHidden: array of Single;
    FWHiddenOutput: array of Single;

    FDeltaOutput: array of Single;
    FDeltaHidden: array of Single;

    FEta: Single;
    FLog: TStrings;
    function GetRandomWeight: Single;
    procedure FeedForward(ASample: PSample); virtual; abstract;
    procedure BackPropagation(ASample: PSample); virtual; abstract;
  public
    constructor Create(ATopology: TTopology); virtual;
    destructor Destroy; override;

    procedure DefineRandomWeights;
    procedure SaveWeights(const AFileName: string);
    procedure LoadWeights(const AFileName: string);
    procedure Learn(ASamplesSet: TSamplesSet);

    property Eta: Single read FEta write FEta;
    property Log: TStrings read FLog write FLog;
  end;

implementation

uses
  System.Math, System.SysUtils, uHelpers, IdGlobal;

{ TNeuralNetworkBase }

constructor TNeuralNetworkBase.Create(ATopology: TTopology);
begin
  FTopology := ATopology;

  SetLength(FNeuronsInput, FTopology.Input + 1); // +1 for BIAS
  SetLength(FNeuronsHidden, FTopology.Hidden + 1); // +1 for BIAS
  SetLength(FNeuronsOutput, FTopology.Output);

  SetLength(FWeightsInputHidden, FTopology.Input + 1, FTopology.Hidden);
  SetLength(FWeightsHiddenOutput, FTopology.Hidden + 1, FTopology.Output);

  SetLength(FWInputHidden, (FTopology.Input + 1) * FTopology.Hidden);
  SetLength(FWHiddenOutput, (FTopology.Hidden + 1) * FTopology.Output);

  SetLength(FDeltaOutput, FTopology.Output);
  SetLength(FDeltaHidden, FTopology.Hidden + 1); // +1 for BIAS

  // BIAS fixed value
  FNeuronsInput[FTopology.Input] := 1;
  FNeuronsHidden[FTopology.Hidden] := 1;

  FLog := nil;
end;

procedure TNeuralNetworkBase.DefineRandomWeights;
var
  i, j, k: Word;
begin
  FLog.Add('DefineRandomWeights');
  Randomize;

  for i := 0 to FTopology.Input {+1 BIAS} do
  begin
    for j := 0 to FTopology.Hidden - 1 do
    begin
      FWeightsInputHidden[i][j] := GetRandomWeight;
      FLog.AddFmt('FWeightsInputHidden[%d][%d]=%.6f', [i, j, FWeightsInputHidden[i][j]]);

      k := i * FTopology.Hidden + j;
      FWInputHidden[k] := FWeightsInputHidden[i][j];
      FLog.AddFmt('FWInputHidden[%d]=%.6f', [k, FWInputHidden[k]]);
    end;
  end;

  for i := 0 to FTopology.Hidden {+1 BIAS} do
  begin
    for j := 0 to FTopology.Output - 1 do
    begin
      FWeightsHiddenOutput[i][j] := GetRandomWeight;
      FLog.AddFmt('FWeightsHiddenOutput[%d][%d]=%.6f', [i, j, FWeightsHiddenOutput[i][j]]);

      k := i * FTopology.Output + j;
      FWHiddenOutput[k] := FWeightsHiddenOutput[i][j];
      FLog.AddFmt('FWHiddenOutput[%d]=%.6f', [k, FWHiddenOutput[k]]);
    end;
  end;
end;

destructor TNeuralNetworkBase.Destroy;
begin
  SetLength(FNeuronsInput, 0);
  SetLength(FNeuronsHidden, 0);
  SetLength(FNeuronsOutput, 0);

  SetLength(FWeightsInputHidden, 0, 0);
  SetLength(FWeightsHiddenOutput, 0, 0);

  SetLength(FDeltaOutput, 0);
  SetLength(FDeltaHidden, 0);

  inherited;
end;

function TNeuralNetworkBase.GetRandomWeight: Single;
begin
  // the weights needs to be initialize with values range -1 to +1
  Result := RandomRange(-1000, 1000) / 1000;
end;

procedure TNeuralNetworkBase.Learn(ASamplesSet: TSamplesSet);
var
  Row: Integer;
  Sample: PSample;
  Error: Single;
begin
  Error := 0;
  for Row := 0 to ASamplesSet.SamplesCount - 1 do
  begin
    Sample := @ASamplesSet.Samples[Row];
    FeedForward(Sample);
    BackPropagation(Sample);

    // TODO : calcular o erro usando o 1/2 * Soma(T - O)^2

    //ReportResults(Sample);
    Error := Error + Abs((Sample^[FTopology.Input]) - Abs(FNeuronsOutput[0]));

//    if Row < 10 then
//      FLog.Strings[Row] := FLog.Strings[Row] + ';' + FloatToStr(Sample^[FTopology.Input] - FNeuronsOutput[0]);

    //FLog.Add('---------- ---------- ----------');
  end;
  Error := Error / ASamplesSet.SamplesCount;
  FLog.Add(FloatToStr(Error));
end;

procedure TNeuralNetworkBase.LoadWeights(const AFileName: string);
var
  Weights: TStringList;
  i, h, o, k: Integer;
  Line: string;
  Value: string;
begin
  // TODO : refatorar para utilizar apenas o peso vetorizado

  try
    Weights := TStringList.Create;
    Weights.LoadFromFile(AFileName);

    for i := 0 to FTopology.Input {+ BIAS} do
    begin
      Line := Weights.Strings[0];
      for h := 0 to FTopology.Hidden - 1 do
      begin
        Value := Fetch(Line, ';');
        FWeightsInputHidden[i][h] := StrToFloat(Value);

        k := i * FTopology.Hidden + h;
        FWInputHidden[k] := FWeightsInputHidden[i][h];
      end;
      Weights.Delete(0);
    end;

    for h := 0 to FTopology.Hidden {+ BIAS} do
    begin
      Line := Weights.Strings[0];
      for o := 0 to FTopology.Output - 1 do
      begin
        Value := Fetch(Line, ';');
        FWeightsHiddenOutput[h][o] := StrToFloat(Value);

        k := h * FTopology.Output + o;
        FWHiddenOutput[k] := FWeightsHiddenOutput[h][o];
      end;
      Weights.Delete(0);
    end;
  finally
    FreeAndNil(Weights);
  end;
end;

procedure TNeuralNetworkBase.SaveWeights(const AFileName: string);
var
  Weights: TStringList;
  Line: string;
  i, h, o: Integer;
begin
  // TODO : refatorar para utilizar apenas o peso vetorizado
  try
    Weights := TStringList.Create;

    for i := 0 to FTopology.Input {+ BIAS} do
    begin
      Line := FloatToStr(FWeightsInputHidden[i][0]);
      for h := 1 to FTopology.Hidden - 1 do
        Line := Line + ';' + FloatToStr(FWeightsInputHidden[i][h]);

      Weights.Add(Line);
    end;

    for h := 0 to FTopology.Hidden {+ BIAS} do
    begin
      Line := FloatToStr(FWeightsHiddenOutput[h][0]);
      for o := 1 to FTopology.Output - 1 do
        Line := Line + ';' + FloatToStr(FWeightsHiddenOutput[h][o]);

      Weights.Add(Line);
    end;

    Weights.SaveToFile(AFileName);
  finally
    FreeAndNil(Weights);
  end;
end;

end.
