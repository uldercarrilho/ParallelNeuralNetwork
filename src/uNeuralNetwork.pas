unit uNeuralNetwork;

interface

uses
  uSamples, System.Classes, System.SysUtils;

type
  TTopology = record
    Input: Word;
    Hidden: Word;
    Output: Word;
  end;

  TNeuralNetwork = class
  private
    FTopology: TTopology;

    // armazena o valor de saída de cada neurônio
    FNeuronsInput: array of Single;
    FNeuronsHidden: array of Single;
    FNeuronsOutput: array of Single;

    // pesos entre os neurônios
    FWeightsInputHidden: array of array of Single;
    FWeightsHiddenOutput: array of array of Single;

    FDeltaOutput: array of Single;
    FDeltaHidden: array of Single;

    FEta: Single;

    FLog: TStrings;

    function GetRandomWeight: Single;
    procedure FeedForward(ASample: PSample);
    procedure BackPropagation(ASample: PSample);
    procedure WriteLog(const Msg: string; Args: array of const);
    procedure ReportResults(ASample: PSample);
  public
    constructor Create(ATopology: TTopology);
    destructor Destroy; override;

    procedure Learn(ASamplesSet: TSamplesSet);
    procedure Tests(ASamplesSet: TSamplesSet);
    procedure DefineRandomWeights;
    procedure SaveWeights(const AFileName: string);
    procedure LoadWeights(const AFileName: string);

    property Eta: Single read FEta write FEta;
    property Log: TStrings read FLog write FLog;
  end;

implementation

uses
  Math, IdGlobal;

{ TNeuralNetwork }

constructor TNeuralNetwork.Create(ATopology: TTopology);
begin
  FTopology := ATopology;

  SetLength(FNeuronsInput, FTopology.Input + 1); // +1 for BIAS
  SetLength(FNeuronsHidden, FTopology.Hidden + 1); // +1 for BIAS
  SetLength(FNeuronsOutput, FTopology.Output);

  // BIAS fixed value
  FNeuronsInput[FTopology.Input] := 1;
  FNeuronsHidden[FTopology.Hidden] := 1;

  SetLength(FWeightsInputHidden, FTopology.Input + 1, FTopology.Hidden);
  SetLength(FWeightsHiddenOutput, FTopology.Hidden + 1, FTopology.Output);

  SetLength(FDeltaOutput, FTopology.Output);
  SetLength(FDeltaHidden, FTopology.Hidden + 1);

  FLog := nil;
end;

destructor TNeuralNetwork.Destroy;
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

procedure TNeuralNetwork.SaveWeights(const AFileName: string);
var
  Weights: TStringList;
  Line: string;
  i, h, o: Integer;
begin
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

procedure TNeuralNetwork.LoadWeights(const AFileName: string);
var
  Weights: TStringList;
  i, h, o: Integer;
  Line: string;
  Value: string;
begin
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
      end;
      Weights.Delete(0);
    end;
  finally
    FreeAndNil(Weights);
  end;
end;

procedure TNeuralNetwork.DefineRandomWeights;
var
  i, j: Word;
begin
  //WriteLog('DefineRandomWeights', []);
  Randomize;

  for i := 0 to FTopology.Input {+1 BIAS} do
  begin
    for j := 0 to FTopology.Hidden - 1 do
    begin
      FWeightsInputHidden[i][j] := GetRandomWeight;
      //WriteLog('FWeightsInputHidden[%d][%d]=%.6f', [i, j, FWeightsInputHidden[i][j]]);
    end;
  end;

  for i := 0 to FTopology.Hidden {+1 BIAS} do
  begin
    for j := 0 to FTopology.Output - 1 do
    begin
      FWeightsHiddenOutput[i][j] := GetRandomWeight;
      //WriteLog('FWeightsHiddenOutput[%d][%d]=%.6f', [i, j, FWeightsHiddenOutput[i][j]]);
    end;
  end;
end;

procedure TNeuralNetwork.FeedForward(ASample: PSample);
var
  i, h, o: Integer;
  Sum: Single;
begin
  //WriteLog('FeedForward', []);
  for i := 0 to FTopology.Input - 1 do
  begin
    FNeuronsInput[i] := ASample^[i];
    //WriteLog('FNeuronsInput[%d]=%.6f', [i, FNeuronsInput[i]]);
  end;
  //WriteLog('FNeuronsInput[%d]=%.6f', [i, FNeuronsInput[i]]);

  for h := 0 to FTopology.Hidden - 1 do
  begin
    Sum := 0;
    for i := 0 to FTopology.Input { +1 BIAS } do
      Sum := Sum + FNeuronsInput[i] * FWeightsInputHidden[i][h];

    FNeuronsHidden[h] := 1 / (1 + Exp(-Sum));
    //WriteLog('FNeuronsHidden[%d]=%.6f', [h, FNeuronsHidden[h]]);
  end;
  //WriteLog('FNeuronsHidden[%d]=%.6f', [h, FNeuronsHidden[h]]);

  for o := 0 to FTopology.Output - 1 do
  begin
    Sum := 0;
    for h := 0 to FTopology.Hidden { +1 BIAS } do
      Sum := Sum + FNeuronsHidden[h] * FWeightsHiddenOutput[h][o];

    FNeuronsOutput[o] := 1 / (1 + Exp(-Sum));
    //WriteLog('FNeuronsOutput[%d]=%.6f', [o, FNeuronsOutput[o]]);
  end;
end;

procedure TNeuralNetwork.BackPropagation(ASample: PSample);
var
  i, h, o, iSample: Word;
  Sum: Single;
begin
  //WriteLog('BackPropagation', []);

  for o := 0 to FTopology.Output - 1 do
  begin
    iSample := FTopology.Input + o;
    FDeltaOutput[o] := FNeuronsOutput[o] * (1 - FNeuronsOutput[o]) * (ASample^[iSample] - FNeuronsOutput[o]);
    //WriteLog('FDeltaOutput[%d]=%.6f', [o, FDeltaOutput[o]]);
  end;

  for h := 0 to FTopology.Hidden { +1 BIAS } do
  begin
    Sum := 0;
    for o := 0 to FTopology.Output - 1 do
      Sum := Sum + (FDeltaOutput[o] * FWeightsHiddenOutput[h][o]);

    FDeltaHidden[h] := FNeuronsHidden[h] * (1 - FNeuronsHidden[h]) * Sum;
    //WriteLog('FDeltaHidden[%d]=%.6f', [h, FDeltaHidden[h]]);
  end;

  for o := 0 to FTopology.Output - 1 do
  begin
    for h := 0 to FTopology.Hidden { +1 BIAS } do
    begin
      FWeightsHiddenOutput[h][o] := FWeightsHiddenOutput[h][o] + FEta * FDeltaOutput[o] * FNeuronsHidden[h];
      //WriteLog('FWeightsHiddenOutput[%d][%d]=%.6f', [h, o, FWeightsHiddenOutput[h][o]]);
    end;
  end;

  for i := 0 to FTopology.Input { +1 BIAS } do
  begin
    for h := 0 to FTopology.Hidden - 1 do
    begin
      FWeightsInputHidden[i][h] := FWeightsInputHidden[i][h] + FEta * FDeltaHidden[h] * FNeuronsInput[i];
      //WriteLog('FWeightsInputHidden[%d][%d]=%.6f', [i, h, FWeightsInputHidden[i][h]]);
    end;
  end;
end;

function TNeuralNetwork.GetRandomWeight: Single;
begin
  // the weights needs to be initialize with values range -1 to +1
  Result := RandomRange(-1000, 1000) / 1000;
end;

procedure TNeuralNetwork.Learn(ASamplesSet: TSamplesSet);
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

    //ReportResults(Sample);
    Error := Error + Abs((Sample^[FTopology.Input]) - Abs(FNeuronsOutput[0]));

//    if Row < 10 then
//      FLog.Strings[Row] := FLog.Strings[Row] + ';' + FloatToStr(Sample^[FTopology.Input] - FNeuronsOutput[0]);

    //FLog.Add('---------- ---------- ----------');
  end;
  Error := Error / ASamplesSet.SamplesCount;
  FLog.Add(FloatToStr(Error));
end;

procedure TNeuralNetwork.ReportResults(ASample: PSample);
var
  Info: string;
  i, iSample: Integer;
begin
  Info := '';
  for i := 0 to FTopology.Input + FTopology.Output - 1 do
    Info := Info + FloatToStr(ASample^[i]) + ';';

  for i := 0 to FTopology.Output - 1 do
  begin
    iSample := FTopology.Input + i;
    Info := Info + ';' + FloatToStr(FNeuronsOutput[i]);
    Info := Info + ';' + FloatToStr(ASample^[iSample] - FNeuronsOutput[i]);
  end;
  FLog.Add(Info);
end;

procedure TNeuralNetwork.Tests(ASamplesSet: TSamplesSet);
var
  Row: Integer;
  Sample: PSample;
  //Error: Single;
begin
  //Error := 0;
  for Row := 0 to ASamplesSet.SamplesCount - 1 do
  begin
    Sample := @ASamplesSet.Samples[Row];
    FeedForward(Sample);

    ReportResults(Sample);
    //Error := Error + Abs((Sample^[FTopology.Input]) - Abs(FNeuronsOutput[0]));
  end;
  //Error := Error / ASamplesSet.SamplesCount;
  //FLog.Add(FloatToStr(Error));
end;

procedure TNeuralNetwork.WriteLog(const Msg: string; Args: array of const);
begin
  FLog.Add(Format(Msg, Args));
end;

end.
