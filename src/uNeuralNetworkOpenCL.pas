unit uNeuralNetworkOpenCL;

interface

uses
  Mitov.OpenCL, System.Classes, uNeuralNetworkBase, uSamples, uTypes;

type
  TNeuralNetworkOpenCL = class(TNeuralNetworkBase)
  protected
    FContext: IOCLContext;
    FCommandQueue: IOCLCommandQueue;
    FKernelMultiply: IOCLKernel;
    FKernelSigmoide: IOCLKernel;
    FKernelDeltaOutput: IOCLKernel;
    FKernelDeltaHidden: IOCLKernel;
    FKernelUpdateWeights: IOCLKernel;
    FKernelParams: IOCLKernel;

    FBufferSamples: IOCLBuffer;

    //FBufferNeuronsInput: IOCLBuffer;
    FBufferNeuronsHidden: IOCLBuffer;
    FBufferNeuronsOutput: IOCLBuffer;

    FBufferWInputHidden: IOCLBuffer;
    FBufferWHiddenOutput: IOCLBuffer;

    FBufferSumInputHidden: IOCLBuffer;
    FBufferSumHiddenOutput: IOCLBuffer;

    FBufferDeltaOutput: IOCLBuffer;
    FBufferDeltaHidden: IOCLBuffer;

    FSumInputHidden: array of Single;
    FSumHiddenOutput: array of Single;

    FSamples: TVector1D;
    FSamplesCount: Cardinal;

    function GetOpenCLSource(const AResourceName: string): string;
    procedure FeedForward(iSample: Cardinal); override;
    procedure BackPropagation(iSample: Cardinal); override;
    procedure CreateBuffers;
  public
    constructor Create(ATopology: TTopology); override;
    destructor Destroy; override;

    procedure BuildKernel;
    procedure WriteBufferToGPU;

    procedure SetSamples(ASamples: TVector1D; ACount: Cardinal);
    procedure Learn(AEpochs: Cardinal); overload;
  end;

implementation

uses
  System.Types, System.SysUtils, uHelpers;

{ TNeuralNetworkOpenCL }

constructor TNeuralNetworkOpenCL.Create(ATopology: TTopology);
begin
  inherited;

  SetLength(FSumInputHidden, (FTopology.Input + 1) * FTopology.Hidden);
  SetLength(FSumHiddenOutput, (FTopology.Hidden + 1) * FTopology.Output);

  BuildKernel;
end;

destructor TNeuralNetworkOpenCL.Destroy;
begin
  SetLength(FSumInputHidden, 0);
  SetLength(FSumHiddenOutput, 0);

  inherited;
end;

procedure TNeuralNetworkOpenCL.BuildKernel;
var
  sOpenCLSource: string;
  APlatform: IOCLPlatform;
  AProgram: IOCLProgram;
begin
  APlatform := TOpenCL.Platforms[0];
  // TODO : avaliar melhor forma de usar apenas a GPU
  APlatform.Devices.Delete(1);

  FContext := TOCLContext.Create(APlatform.Devices);

  sOpenCLSource := GetOpenCLSource('KernelNeuralNetwork');
  AProgram := FContext.CreateProgramFromSource(sOpenCLSource);
  try
    AProgram.Build();
  except
    on E: Exception do
    begin
      E.Message := E.Message + ' | ' + AProgram.BuildLog[APlatform.Devices[0]];
      raise E;
    end;
  end;

  FKernelMultiply := AProgram.Kernel['multiply'];
  FKernelSigmoide := AProgram.Kernel['sigmoide'];
  FKernelDeltaOutput := AProgram.Kernel['calculateDeltaOutput'];
  FKernelDeltaHidden := AProgram.Kernel['calculateDeltaHidden'];
  FKernelUpdateWeights := AProgram.Kernel['updateWeights'];
  FKernelParams := AProgram.Kernel['params'];

  FCommandQueue := FContext.CreateCommandQueue();
end;

function TNeuralNetworkOpenCL.GetOpenCLSource(const AResourceName: string): string;
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

procedure TNeuralNetworkOpenCL.SetSamples(ASamples: TVector1D; ACount: Cardinal);
begin
  FSamples := ASamples;
  FSamplesCount := ACount;
end;

procedure TNeuralNetworkOpenCL.Learn(AEpochs: Cardinal);
var
  i: Integer;
  j: Integer;
begin
  CreateBuffers;
  WriteBufferToGPU;

  for i := 1 to AEpochs do
  begin
    for j := 0 to FSamplesCount - 1 do
    begin
      FeedForward(j);
      BackPropagation(j);
    end;
  end;
  FCommandQueue.Finish;
end;

procedure TNeuralNetworkOpenCL.CreateBuffers;
var
  SampleSize: Word;
  BufferSize: Cardinal;
begin
  // TODO : improve memory access

  SampleSize := FTopology.Input + 1 + FTopology.Output; // +1 for BIAS
  BufferSize := SampleSize * FSamplesCount * SizeOf(Single);

  FBufferSamples := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], BufferSize, @FSamples[0]);

  //FBufferNeuronsInput := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (FTopology.Input + 1) * SizeOf(Single), @FNeuronsInput[0]); // +1 for BIAS
  FBufferNeuronsHidden := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (FTopology.Hidden + 1) * SizeOf(Single), @FNeuronsHidden[0]); // +1 for BIAS
  FBufferNeuronsOutput := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], FTopology.Output * SizeOf(Single), @FNeuronsOutput[0]);

  FBufferWInputHidden := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (FTopology.Input + 1) * FTopology.Hidden * SizeOf(Single), @FWeights1DInputHidden[0]);
  FBufferWHiddenOutput := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (FTopology.Hidden + 1) * FTopology.Output * SizeOf(Single), @FWeights1DHiddenOutput[0]);

  FBufferSumInputHidden := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (FTopology.Input + 1) * FTopology.Hidden * SizeOf(Single), @FSumInputHidden[0]);
  FBufferSumHiddenOutput := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (FTopology.Hidden + 1) * FTopology.Output * SizeOf(Single), @FSumHiddenOutput[0]);

  FBufferDeltaOutput := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (FTopology.Output) * SizeOf(Single), @FDeltaOutput[0]);
  FBufferDeltaHidden := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (FTopology.Hidden + 1) * SizeOf(Single), @FDeltaHidden[0]); // +1 for BIAS
end;

procedure TNeuralNetworkOpenCL.WriteBufferToGPU;
begin
  FCommandQueue.EnqueueWriteBuffer(FBufferSamples, True, @FSamples[0]);

  //FCommandQueue.EnqueueWriteBuffer(FBufferNeuronsInput, True, @FNeuronsInput[0]);
  FCommandQueue.EnqueueWriteBuffer(FBufferNeuronsHidden, True, @FNeuronsHidden[0]);
  FCommandQueue.EnqueueWriteBuffer(FBufferNeuronsOutput, True, @FNeuronsOutput[0]);

  FCommandQueue.EnqueueWriteBuffer(FBufferWInputHidden, True, @FWeights1DInputHidden[0]);
  FCommandQueue.EnqueueWriteBuffer(FBufferWHiddenOutput, True, @FWeights1DHiddenOutput[0]);

  FCommandQueue.EnqueueWriteBuffer(FBufferSumInputHidden, True, @FSumInputHidden[0]);
  FCommandQueue.EnqueueWriteBuffer(FBufferSumHiddenOutput, True, @FSumHiddenOutput[0]);

  FCommandQueue.EnqueueWriteBuffer(FBufferDeltaOutput, True, @FDeltaOutput[0]);
  FCommandQueue.EnqueueWriteBuffer(FBufferDeltaHidden, True, @FDeltaHidden[0]);

  FCommandQueue.Finish;
end;

procedure TNeuralNetworkOpenCL.FeedForward(iSample: Cardinal);
var
//  i: Integer;
  InputOffSet: Cardinal;
begin
//  FLog.Add('FeedForward');

//  for i := 0 to FTopology.Input - 1 do
//  begin
//    FNeuronsInput[i] := FSamplesSet.Samples[iSample][i];
//    FLog.AddFmt('FNeuronsInput[%d]=%.6f', [i, FNeuronsInput[i]]);
//  end;
//  FLog.Add('');
//  FCommandQueue.EnqueueWriteBuffer(FBufferNeuronsInput, True, @FNeuronsInput[0]);


  {$REGION 'Calcular ativação INPUT --> HIDDEN'}
  InputOffSet := iSample * (FTopology.Input + 1 + FTopology.Output); // +1 for BIAS

  FKernelMultiply.Arguments[0].Access.SetBuffer(FBufferSamples);
  FKernelMultiply.Arguments[1].Access.SetBuffer(FBufferWInputHidden);
  FKernelMultiply.Arguments[2].Access.SetBuffer(FBufferSumInputHidden);
  FKernelMultiply.Arguments[3].Access.SetValue<Cardinal>(InputOffSet);
  FKernelMultiply.Arguments[4].Access.SetValue<Cardinal>(FTopology.Input + 1); // +1 for BIAS
  FKernelMultiply.Arguments[5].Access.SetValue<Cardinal>(FTopology.Hidden);

  FCommandQueue.EnqueueNDRangeKernel(FKernelMultiply, TOCLGlobalDimensions.Create([FTopology.Input + 1, FTopology.Hidden]));  // +1 for BIAS
  //FCommandQueue.Finish;
  //FCommandQueue.EnqueueReadBuffer(FBufferSumInputHidden, True, @FSumInputHidden[0]);

//  for i := 0 to (FTopology.Input + 1) * FTopology.Hidden - 1 do
//    FLog.Add(Format('FSumInputHidden[%d] = %.6f', [i, FSumInputHidden[i]]));
//  FLog.Add('');

  FKernelSigmoide.Arguments[0].Access.SetBuffer(FBufferSumInputHidden);
  FKernelSigmoide.Arguments[1].Access.SetBuffer(FBufferNeuronsHidden);
  FKernelSigmoide.Arguments[2].Access.SetValue<Cardinal>(FTopology.Input + 1); // +1 for BIAS
  FKernelSigmoide.Arguments[3].Access.SetValue<Cardinal>(FTopology.Hidden);

  FCommandQueue.EnqueueNDRangeKernel(FKernelSigmoide, TOCLGlobalDimensions.Create([FTopology.Hidden]));
  //FCommandQueue.Finish;
  //FCommandQueue.EnqueueReadBuffer(FBufferNeuronsHidden, True, @FNeuronsHidden[0]);

//  for i := 0 to FTopology.Hidden - 1 do
//    FLog.Add(Format('FNeuronsHidden[%d] = %.6f', [i, FNeuronsHidden[i]]));
//  FLog.Add('');
  {$ENDREGION}


  {$REGION 'Calcular ativação HIDDEN --> OUTPUT'}
  FKernelMultiply.Arguments[0].Access.SetBuffer(FBufferNeuronsHidden);
  FKernelMultiply.Arguments[1].Access.SetBuffer(FBufferWHiddenOutput);
  FKernelMultiply.Arguments[2].Access.SetBuffer(FBufferSumHiddenOutput);
  FKernelMultiply.Arguments[3].Access.SetValue<Cardinal>(0);
  FKernelMultiply.Arguments[4].Access.SetValue<Cardinal>(FTopology.Hidden + 1); // +1 for BIAS
  FKernelMultiply.Arguments[5].Access.SetValue<Cardinal>(FTopology.Output);

  FCommandQueue.EnqueueNDRangeKernel(FKernelMultiply, TOCLGlobalDimensions.Create([FTopology.Hidden + 1, FTopology.Output]));
  //FCommandQueue.Finish;
  //FCommandQueue.EnqueueReadBuffer(FBufferSumHiddenOutput, True, @FSumHiddenOutput[0]);

//  for i := 0 to (FTopology.Hidden + 1) * FTopology.Output - 1 do
//    FLog.Add(Format('FSumHiddenOutput[%d] = %.6f', [i, FSumHiddenOutput[i]]));
//  FLog.Add('');

  FKernelSigmoide.Arguments[0].Access.SetBuffer(FBufferSumHiddenOutput);
  FKernelSigmoide.Arguments[1].Access.SetBuffer(FBufferNeuronsOutput);
  FKernelSigmoide.Arguments[2].Access.SetValue<Cardinal>(FTopology.Hidden + 1); // +1 for BIAS
  FKernelSigmoide.Arguments[3].Access.SetValue<Cardinal>(FTopology.Output);

  FCommandQueue.EnqueueNDRangeKernel(FKernelSigmoide, TOCLGlobalDimensions.Create([FTopology.Output]));
  //FCommandQueue.Finish;
  //FCommandQueue.EnqueueReadBuffer(FBufferNeuronsOutput, True, @FNeuronsOutput[0]);

//  for i := 0 to FTopology.Output - 1 do
//    FLog.Add(Format('FNeuronsOutput[%d] = %.6f', [i, FNeuronsOutput[i]]));
//  FLog.Add('');
  {$ENDREGION}
end;

procedure TNeuralNetworkOpenCL.BackPropagation(iSample: Cardinal);
var
  //i, iOutput: Integer;
  OutputOffset: Cardinal;
  NeuronOffset: Cardinal;
begin
//  FLog.Add('BackPropagation');
//  for i := 0 to FTopology.Output - 1 do
//  begin
//    iOutput := FTopology.Input + i;
//    FSampleOutput[i] := FSamplesSet.Samples[iSample][iOutput];
//
//    FLog.AddFmt('FSampleOutput[%d]=%.6f', [i, FSampleOutput[i]]);
//  end;
//  FLog.Add('');
//  FCommandQueue.EnqueueWriteBuffer(FBufferSampleOutput, True, @FSampleOutput[0]);

  {$REGION 'Delta OUTPUT'}
  OutputOffset := iSample * (FTopology.Input + 1 + FTopology.Output) + FTopology.Input; // +1 for BIAS

  FKernelDeltaOutput.Arguments[0].Access.SetBuffer(FBufferNeuronsOutput);
  FKernelDeltaOutput.Arguments[1].Access.SetBuffer(FBufferSamples);
  FKernelDeltaOutput.Arguments[2].Access.SetBuffer(FBufferDeltaOutput);
  FKernelDeltaOutput.Arguments[3].Access.SetValue<Cardinal>(FTopology.Output);
  FKernelDeltaOutput.Arguments[4].Access.SetValue<Cardinal>(OutputOffset);

  FCommandQueue.EnqueueNDRangeKernel(FKernelDeltaOutput, TOCLGlobalDimensions.Create([FTopology.Output]));
  //FCommandQueue.Finish;
  //FCommandQueue.EnqueueReadBuffer(FBufferDeltaOutput, True, @FDeltaOutput[0]);

//  for i := 0 to FTopology.Output - 1 do
//    FLog.Add(Format('DeltaOutput[%d] = %.6f', [i, FDeltaOutput[i]]));
//  FLog.Add('');
  {$ENDREGION}

  {$REGION 'Delta HIDDEN'}
  FKernelDeltaHidden.Arguments[0].Access.SetBuffer(FBufferNeuronsHidden);
  FKernelDeltaHidden.Arguments[1].Access.SetBuffer(FBufferDeltaOutput);
  FKernelDeltaHidden.Arguments[2].Access.SetBuffer(FBufferWHiddenOutput);
  FKernelDeltaHidden.Arguments[3].Access.SetBuffer(FBufferDeltaHidden);
  FKernelDeltaHidden.Arguments[4].Access.SetValue<Cardinal>(FTopology.Hidden + 1); // +1 for BIAS
  FKernelDeltaHidden.Arguments[5].Access.SetValue<Cardinal>(FTopology.Output);

  FCommandQueue.EnqueueNDRangeKernel(FKernelDeltaHidden, TOCLGlobalDimensions.Create([FTopology.Hidden + 1]));  // +1 for BIAS
  //FCommandQueue.Finish;
  //FCommandQueue.EnqueueReadBuffer(FBufferDeltaHidden, True, @FDeltaHidden[0]);

//  for i := 0 to FTopology.Hidden {+1 for BIAS} do
//    FLog.Add(Format('DeltaHidden[%d] = %.6f', [i, FDeltaHidden[i]]));
//  FLog.Add('');
  {$ENDREGION}

  {$REGION 'Update Weights HIDDEN --> OUTPUT'}
  NeuronOffset := 0;
  FKernelUpdateWeights.Arguments[0].Access.SetBuffer(FBufferWHiddenOutput);
  FKernelUpdateWeights.Arguments[1].Access.SetBuffer(FBufferNeuronsHidden);
  FKernelUpdateWeights.Arguments[2].Access.SetBuffer(FBufferDeltaOutput);
  FKernelUpdateWeights.Arguments[3].Access.SetValue<Single>(NeuronOffset);
  FKernelUpdateWeights.Arguments[4].Access.SetValue<Cardinal>(FTopology.Hidden + 1); // +1 for BIAS
  FKernelUpdateWeights.Arguments[5].Access.SetValue<Cardinal>(FTopology.Output);
  FKernelUpdateWeights.Arguments[6].Access.SetValue<Single>(ETA);

  FCommandQueue.EnqueueNDRangeKernel(FKernelUpdateWeights, TOCLGlobalDimensions.Create([FTopology.Hidden + 1, FTopology.Output]));
  //FCommandQueue.Finish;
  //FCommandQueue.EnqueueReadBuffer(FBufferWHiddenOutput, True, @FWHiddenOutput[0]);

//  for i := 0 to ((FTopology.Hidden + 1) * FTopology.Output) - 1 do
//    FLog.Add(Format('FWHiddenOutput[%d] = %.6f', [i, FWHiddenOutput[i]]));
//  FLog.Add('');
  {$ENDREGION}

  {$REGION 'Update Weights INPUT --> HIDDEN'}
  NeuronOffset := iSample * (FTopology.Input + 1 + FTopology.Output); // +1 for BIAS
  FKernelUpdateWeights.Arguments[0].Access.SetBuffer(FBufferWInputHidden);
  FKernelUpdateWeights.Arguments[1].Access.SetBuffer(FBufferSamples);
  FKernelUpdateWeights.Arguments[2].Access.SetBuffer(FBufferDeltaHidden);
  FKernelUpdateWeights.Arguments[3].Access.SetValue<Cardinal>(NeuronOffset);
  FKernelUpdateWeights.Arguments[4].Access.SetValue<Cardinal>(FTopology.Input + 1); // +1 for BIAS
  FKernelUpdateWeights.Arguments[5].Access.SetValue<Cardinal>(FTopology.Hidden);
  FKernelUpdateWeights.Arguments[6].Access.SetValue<Single>(ETA);

  FCommandQueue.EnqueueNDRangeKernel(FKernelUpdateWeights, TOCLGlobalDimensions.Create([FTopology.Input + 1, FTopology.Hidden]));
  //FCommandQueue.Finish;
  //FCommandQueue.EnqueueReadBuffer(FBufferWInputHidden, True, @FWInputHidden[0]);

//  for i := 0 to ((FTopology.Input + 1) * FTopology.Hidden) - 1 do
//    FLog.Add(Format('FWInputHidden[%d] = %.6f', [i, FWInputHidden[i]]));
//  FLog.Add('');
  {$ENDREGION}
end;

end.
