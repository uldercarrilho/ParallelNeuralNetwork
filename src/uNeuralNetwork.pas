unit uNeuralNetwork;

interface

uses
  uSamples, uNeuralNetworkBase, System.Classes, System.SysUtils, Mitov.OpenCL;

type
  PVector1D = ^TVector1D;
  PVector2D = ^TVector2D;

  TVector1D = array of Single;
  TVector2D = array of TVector1D;

  TNeuralNetwork = class(TNeuralNetworkBase)
  private
    procedure ReportResults(ASample: PSample);
    procedure Compute(ANeuronsIN, ANeuronsOUT: PVector1D; AWeights: PVector2D; const ASizeIN, ASizeOUT: Word);
  protected
    procedure FeedForward(ASample: PSample); override;
    procedure BackPropagation(ASample: PSample); override;
  public
    procedure Tests(ASamplesSet: TSamplesSet);
  end;

implementation

uses
  Math, IdGlobal, Winapi.Windows;

{ TNeuralNetwork }

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
  i: Integer;
//  Sum: Single;
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

end.
