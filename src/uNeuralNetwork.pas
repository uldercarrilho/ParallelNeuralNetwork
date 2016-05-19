unit uNeuralNetwork;

interface

uses
  uSamples, System.Classes, System.SysUtils, Mitov.OpenCL;

type
  TTopology = record
    Input: Word;
    Hidden: Word;
    Output: Word;
  end;

  PVector1D = ^TVector1D;
  PVector2D = ^TVector2D;

  TVector1D = array of Single;
  TVector2D = array of TVector1D;

  TNeuralNetwork = class
  private
    FTopology: TTopology;

    // armazena o valor de saída de cada neurônio
    FNeuronsInput: TVector1D;
    FNeuronsHidden: TVector1D;
    FNeuronsOutput: TVector1D;

    // pesos entre os neurônios
    FWeightsInputHidden: TVector2D;
    FWeightsHiddenOutput: TVector2D;

    FDeltaOutput: TVector1D;
    FDeltaHidden: TVector1D;

    FEta: Single;

    FLog: TStrings;

    FContext: IOCLContext;
    FCommandQueue: IOCLCommandQueue;
    FKernel: IOCLKernel;

    function GetRandomWeight: Single;
    procedure FeedForward(ASample: PSample);
    procedure BackPropagation(ASample: PSample);
    procedure WriteLog(const Msg: string; Args: array of const);
    procedure ReportResults(ASample: PSample);
    function GetOpenCLSource(const AResourceName: string): string;
    procedure BuildKernel;
    procedure Compute(ANeuronsIN, ANeuronsOUT: PVector1D; AWeights: PVector2D; const ASizeIN, ASizeOUT: Word);
  public
    constructor Create(ATopology: TTopology);
    destructor Destroy; override;

    procedure Learn(ASamplesSet: TSamplesSet);
    procedure Tests(ASamplesSet: TSamplesSet);
    procedure DefineRandomWeights;
    procedure SaveWeights(const AFileName: string);
    procedure LoadWeights(const AFileName: string);
    procedure LoadGPU(ASamplesSet: TSamplesSet);

    property Eta: Single read FEta write FEta;
    property Log: TStrings read FLog write FLog;
  end;

implementation

uses
  Math, IdGlobal, Winapi.Windows;

{ TNeuralNetwork }

procedure TNeuralNetwork.BuildKernel;
var
  sOpenCLSource: string;
  APlatform: IOCLPlatform;
  AProgram: IOCLProgram;
  i: Integer;
begin
  sOpenCLSource := GetOpenCLSource('KernelNeuralNetwork');

  APlatform := TOpenCL.Platforms[0];
  FContext := TOCLContext.Create(APlatform.Devices);
  AProgram := FContext.CreateProgramFromSource(sOpenCLSource);
  try
    AProgram.Build();
  except on E: Exception do
    FLog.Add('BuildLog: ' + AProgram.BuildLog[APlatform.Devices[0]]);
  end;

  FKernel := AProgram.Kernels[2];
  FCommandQueue := FContext.CreateCommandQueue();
end;

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

  BuildKernel;
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

procedure TNeuralNetwork.LoadGPU(ASamplesSet: TSamplesSet);
var
  Resultado: array of Single;
  Amostras: IOCLBuffer;
  Buffer: IOCLBuffer;
  BufferSize: Cardinal;
  i: Integer;
begin
  BufferSize := ASamplesSet.SamplesCount * ASamplesSet.SampleSize;
  SetLength(Resultado, BufferSize);

  Amostras := FContext.CreateBuffer([TOCLMemoryFlag.ReadOnly, TOCLMemoryFlag.CopyHostPtr], BufferSize * SizeOf(Single), @(ASamplesSet.FRaw[0]));
  Buffer := FContext.CreateBuffer([TOCLMemoryFlag.WriteOnly], BufferSize * SizeOf(Single));

  // Create OpenCL buffers
//  ACLBuffer1 := FContext.CreateBuffer([TOCLMemoryFlag.ReadOnly, TOCLMemoryFlag.CopyHostPtr], BUFFER_SIZE * SizeOf(Single), @FBuffer1[0]);
//  ACLBuffer2 := FContext.CreateBuffer([TOCLMemoryFlag.ReadOnly, TOCLMemoryFlag.CopyHostPtr], BUFFER_SIZE * SizeOf(Single), @FBuffer2[0]);
//  ACLBuffer3 := FContext.CreateBuffer([TOCLMemoryFlag.ReadOnly, TOCLMemoryFlag.CopyHostPtr], BUFFER_SIZE * SizeOf(Single), @FBuffer3[0]);
//  ACLResultBuffer := FContext.CreateBuffer( [TOCLMemoryFlag.WriteOnly], BUFFER_SIZE * SizeOf(Single) );
//  GlobalBuffer := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite], 10, nil);

  // Set the Kernel arguments
  FKernel.Arguments[0].Access.SetBuffer(Amostras);
  FKernel.Arguments[1].Access.SetBuffer(Buffer);
  FKernel.Arguments[2].Access.SetValue<Cardinal>(ASamplesSet.InputSize);
  FKernel.Arguments[3].Access.SetValue<Cardinal>(ASamplesSet.OutputSize);
  FKernel.Arguments[4].Access.SetValue<Cardinal>(ASamplesSet.SamplesCount);

  FCommandQueue.EnqueueNDRangeKernel(FKernel, TOCLGlobalDimensions.Create([ASamplesSet.SamplesCount]));

  FCommandQueue.EnqueueReadBuffer(Buffer, True, @Resultado[0]);

  for i := 0 to ASamplesSet.SamplesCount - 1 do
    Log.Add(IntToStr(i) + ' = ' + FloatToStr(Resultado[i]));
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

procedure TNeuralNetwork.Compute(ANeuronsIN, ANeuronsOUT: PVector1D; AWeights: PVector2D; const ASizeIN, ASizeOUT: Word);
var
  i, o: Word;
  Sum: Single;
begin
  for o := 0 to ASizeOUT - 1 do
  begin
    Sum := 0;
    for i := 0 to ASizeIN - 1 do
      Sum := Sum + ANeuronsIN^[i] * AWeights^[i][o];

    ANeuronsOUT^[o] := 1 / (1 + Exp(-Sum));
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

  Compute(@FNeuronsInput, @FNeuronsHidden, @FWeightsInputHidden, FTopology.Input + 1, FTopology.Hidden);
  (*
  for h := 0 to FTopology.Hidden - 1 do
  begin
    Sum := 0;
    for i := 0 to FTopology.Input { +1 BIAS } do
      Sum := Sum + FNeuronsInput[i] * FWeightsInputHidden[i][h];

    FNeuronsHidden[h] := 1 / (1 + Exp(-Sum));
    //WriteLog('FNeuronsHidden[%d]=%.6f', [h, FNeuronsHidden[h]]);
  end;
  //WriteLog('FNeuronsHidden[%d]=%.6f', [h, FNeuronsHidden[h]]);
  *)

  Compute(@FNeuronsHidden, @FNeuronsOutput, @FWeightsHiddenOutput, FTopology.Hidden + 1, FTopology.Output);
  (*
  for o := 0 to FTopology.Output - 1 do
  begin
    Sum := 0;
    for h := 0 to FTopology.Hidden { +1 BIAS } do
      Sum := Sum + FNeuronsHidden[h] * FWeightsHiddenOutput[h][o];

    FNeuronsOutput[o] := 1 / (1 + Exp(-Sum));
    //WriteLog('FNeuronsOutput[%d]=%.6f', [o, FNeuronsOutput[o]]);
  end;
  *)
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

  for h := 0 to FTopology.Hidden { +1 BIAS } do
  begin
    for o := 0 to FTopology.Output - 1 do
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

function TNeuralNetwork.GetOpenCLSource(const AResourceName: string): string;
var
  ResStream: TResourceStream;
  slResource: TStringList;
begin
  ResStream := TResourceStream.Create(hInstance, AResourceName, RT_RCDATA);
  slResource := TStringList.Create;
  try
    slResource.LoadFromStream(ResStream);
    Result := slResource.Text;
  finally
    ResStream.Free;
    slResource.Free;
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
