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
    procedure ReportResults(iSample: Cardinal);
    procedure Compute(ANeuronsIN, ANeuronsOUT: PVector1D; AWeights: PVector2D; const ASizeIN, ASizeOUT: Word);
  protected
    procedure FeedForward(iSample: Cardinal); override;
    procedure BackPropagation(iSample: Cardinal); override;
  public
    procedure Tests(ASamplesSet: TSamplesSet);
  end;

implementation

uses
  Math, IdGlobal, Winapi.Windows, uHelpers, System.Threading;

{ TNeuralNetwork }

procedure TNeuralNetwork.Compute(ANeuronsIN, ANeuronsOUT: PVector1D; AWeights: PVector2D; const ASizeIN, ASizeOUT: Word);
var
  i, o: Word;
  Sum: Extended;
begin
  //TParallel.For(0, ASizeOUT - 1, procedure(o: Integer)
  for o := 0 to ASizeOUT - 1 do
  //var i: Word;
  begin
    Sum := 0;
    for i := 0 to ASizeIN - 1 do
      Sum := Sum + ANeuronsIN^[i] * AWeights^[i][o];

    try
      ANeuronsOUT^[o] := 1 / (1 + Exp(-Sum));
    except
      ANeuronsOUT^[o] := 0;
    end;
  end;
end;

procedure TNeuralNetwork.FeedForward(iSample: Cardinal);
var
  i: Integer;
//  Sum: Single;
begin
  ////FLog.Add('FeedForward');

  //TParallel.For(0, FTopology.Input - 1, procedure(i: Integer)
  for i := 0 to FTopology.Input - 1 do
  begin
    FNeuronsInput[i] := FSamplesSet.Samples[iSample][i];
    //FLog.AddFmt('FNeuronsInput[%d]=%.6f', [i, FNeuronsInput[i]]);
  end;
  //FLog.Add('');

  Compute(@FNeuronsInput, @FNeuronsHidden, @FWeightsInputHidden, FTopology.Input + 1, FTopology.Hidden);

  //for i := 0 to FTopology.Hidden - 1 do
    //FLog.AddFmt('FNeuronsHidden[%d] = %.6f', [i, FNeuronsHidden[i]]);
  //FLog.Add('');

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

  //for i := 0 to FTopology.Output - 1 do
    //FLog.AddFmt('FNeuronsOutput[%d] = %.6f', [i, FNeuronsOutput[i]]);
  //FLog.Add('');

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

procedure TNeuralNetwork.BackPropagation(iSample: Cardinal);
var
  i, h, o, iOutput: Word;
  Sum: Single;
begin
  //FLog.Add('BackPropagation');

  //TParallel.For(0, FTopology.Output - 1, procedure(o: Integer)
  for o := 0 to FTopology.Output - 1 do
  begin
    iOutput := FTopology.Input + o;
    FDeltaOutput[o] := FNeuronsOutput[o] * (1 - FNeuronsOutput[o]) * (FSamplesSet.Samples[iSample][iOutput] - FNeuronsOutput[o]);
    //FLog.AddFmt('FDeltaOutput[%d]=%.6f', [o, FDeltaOutput[o]]);
  end;
  //FLog.Add('');

  //TParallel.For(0, FTopology.Hidden, procedure(h: Integer)
  for h := 0 to FTopology.Hidden { +1 BIAS } do
  //var o: Integer;
  begin
    Sum := 0;
    for o := 0 to FTopology.Output - 1 do
      Sum := Sum + (FDeltaOutput[o] * FWeightsHiddenOutput[h][o]);

    FDeltaHidden[h] := FNeuronsHidden[h] * (1 - FNeuronsHidden[h]) * Sum;
    //FLog.AddFmt('FDeltaHidden[%d]=%.6f', [h, FDeltaHidden[h]]);
  end;
  //FLog.Add('');

  //TParallel.For(0, FTopology.Hidden, procedure(h: Integer)
  for h := 0 to FTopology.Hidden { +1 BIAS } do
  //var o: Integer;
  begin
    for o := 0 to FTopology.Output - 1 do
    begin
      FWeightsHiddenOutput[h][o] := FWeightsHiddenOutput[h][o] + FEta * FDeltaOutput[o] * FNeuronsHidden[h];
      //FLog.AddFmt('FWeightsHiddenOutput[%d][%d]=%.6f', [h, o, FWeightsHiddenOutput[h][o]]);
    end;
  end;
  //FLog.Add('');

  //TParallel.For(0, FTopology.Input, procedure(i: Integer)
  for i := 0 to FTopology.Input { +1 BIAS } do
  //var h: Integer;
  begin
    for h := 0 to FTopology.Hidden - 1 do
    begin
      FWeightsInputHidden[i][h] := FWeightsInputHidden[i][h] + FEta * FDeltaHidden[h] * FNeuronsInput[i];
      //FLog.AddFmt('FWeightsInputHidden[%d][%d]=%.6f', [i, h, FWeightsInputHidden[i][h]]);
    end;
  end;
  //FLog.Add('');
end;

procedure TNeuralNetwork.ReportResults(iSample: Cardinal);
var
  Info: string;
  i, iOutput: Integer;
begin
  Info := '';
  for i := 0 to FTopology.Input + FTopology.Output - 1 do
    Info := Info + FloatToStr(FSamplesSet.Samples[iSample][i]) + ';';

  for i := 0 to FTopology.Output - 1 do
  begin
    iOutput := FTopology.Input + i;
    Info := Info + ';' + FloatToStr(FNeuronsOutput[i]);
    Info := Info + ';' + FloatToStr(FSamplesSet.Samples[iSample][iOutput] - FNeuronsOutput[i]);
  end;
  //FLog.Add(Info);
end;

procedure TNeuralNetwork.Tests(ASamplesSet: TSamplesSet);
var
  Row: Integer;
  //Sample: PSample;
  //Error: Single;
begin
  //Error := 0;
  for Row := 0 to ASamplesSet.SamplesCount - 1 do
  begin
    //Sample := @ASamplesSet.Samples[Row];
    FeedForward(Row);

    ReportResults(Row);
    //Error := Error + Abs((Sample^[FTopology.Input]) - Abs(FNeuronsOutput[0]));
  end;
  //Error := Error / ASamplesSet.SamplesCount;
  //FLog.Add(FloatToStr(Error));
end;

end.
