unit uNeuralNetworkOpenCL;

interface

uses
  Mitov.OpenCL, System.Classes, uNeuralNetworkBase, uSamples;

type
  TNeuralNetworkOpenCL = class(TNeuralNetworkBase)
  private
    FContext: IOCLContext;
    FCommandQueue: IOCLCommandQueue;
    FKernelMultiply: IOCLKernel;
    FKernelSigmoide: IOCLKernel;
    FKernelDeltaOutput: IOCLKernel;
    FKernelDeltaHidden: IOCLKernel;
    FKernelUpdateWeights: IOCLKernel;

    FBufferNeuronsInput: IOCLBuffer;
    FBufferNeuronsHidden: IOCLBuffer;
    FBufferNeuronsOutput: IOCLBuffer;

    FBufferWInputHidden: IOCLBuffer;
    FBufferWHiddenOutput: IOCLBuffer;

    FBufferDeltaOutput: IOCLBuffer;
    FBufferDeltaHidden: IOCLBuffer;


    FSumInputHidden: array of Single;
    FSumHiddenOutput: array of Single;

    FBufferSumInputHidden: IOCLBuffer;
    FBufferSumHiddenOutput: IOCLBuffer;

    FSampleOutput: array of Single;
    FBufferSampleOutput: IOCLBuffer;

    function GetOpenCLSource(const AResourceName: string): string;
  protected
    procedure FeedForward(ASample: PSample); override;
    procedure BackPropagation(ASample: PSample); override;
  public
    constructor Create(ATopology: TTopology); override;
    procedure BuildKernel;
    procedure Multiply;
    procedure DeltaOutput;
    procedure DeltaHidden;
    procedure UpdateWeights;
  end;

implementation

uses
  System.Types, System.SysUtils, uHelpers;

{ TNeuralNetworkOpenCL }


procedure TNeuralNetworkOpenCL.DeltaHidden;
const
  HIDDEN_SIZE = 3;
  OUTPUT_SIZE = 2;
  DELTAO_SIZE = OUTPUT_SIZE;
  WEIGHT_SIZE = HIDDEN_SIZE * OUTPUT_SIZE;
  RESULT_SIZE = HIDDEN_SIZE;
var
  Hiddens: array[0..HIDDEN_SIZE - 1] of Single;
  DeltaOu: array[0..DELTAO_SIZE - 1] of Single;
  Weights: array[0..WEIGHT_SIZE - 1] of Single;
  Results: array[0..RESULT_SIZE - 1] of Single;

  BufferHiddens: IOCLBuffer;
  BufferDeltaOu: IOCLBuffer;
  BufferWeights: IOCLBuffer;
  BufferResults: IOCLBuffer;

  i: Integer;
begin
  Hiddens[0] := 0.2;
  Hiddens[1] := 0.3;
  Hiddens[2] := 0.4;

  DeltaOu[0] := -0.032;
  DeltaOu[1] := +0.147;

  Weights[0] := 1;
  Weights[1] := 2;
  Weights[2] := 3;
  Weights[3] := 4;
  Weights[4] := 5;
  Weights[5] := 6;

  Results[0] := 0;
  Results[1] := 0;
  Results[2] := 0;

  BufferHiddens := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], HIDDEN_SIZE * SizeOf(Single), @Hiddens[0]);
  BufferDeltaOu := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], DELTAO_SIZE * SizeOf(Single), @DeltaOu[0]);
  BufferWeights := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], WEIGHT_SIZE * SizeOf(Single), @Weights[0]);
  BufferResults := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], RESULT_SIZE * SizeOf(Single), @Results[0]);

  FKernelDeltaHidden.Arguments[0].Access.SetBuffer(BufferHiddens);
  FKernelDeltaHidden.Arguments[1].Access.SetBuffer(BufferDeltaOu);
  FKernelDeltaHidden.Arguments[2].Access.SetBuffer(BufferWeights);
  FKernelDeltaHidden.Arguments[3].Access.SetBuffer(BufferResults);
  FKernelDeltaHidden.Arguments[4].Access.SetValue<Cardinal>(HIDDEN_SIZE);
  FKernelDeltaHidden.Arguments[5].Access.SetValue<Cardinal>(OUTPUT_SIZE);

  FCommandQueue.EnqueueNDRangeKernel(FKernelDeltaHidden, TOCLGlobalDimensions.Create([HIDDEN_SIZE]));
  FCommandQueue.EnqueueReadBuffer(BufferResults, True, @Results[0]);

  for i := 0 to RESULT_SIZE - 1 do
    FLog.Add(Format('DeltaHidden[%d] = %.6f', [i, Results[i]]));

  FLog.Add('');
end;

procedure TNeuralNetworkOpenCL.DeltaOutput;
const
  OUTPUT_SIZE = 2;
  SAMPLE_SIZE = OUTPUT_SIZE;
  RESULT_SIZE = OUTPUT_SIZE;
var
  Outputs: array[0..OUTPUT_SIZE - 1] of Single;
  Samples: array[0..SAMPLE_SIZE - 1] of Single;
  Results: array[0..RESULT_SIZE - 1] of Single;

  BufferOutputs: IOCLBuffer;
  BufferSamples: IOCLBuffer;
  BufferResults: IOCLBuffer;

  i: Integer;
begin
  Outputs[0] := 0.2;
  Outputs[1] := 0.3;

  Samples[0] := 0;
  Samples[1] := 1;

  Results[0] := 0;
  Results[1] := 0;

  BufferOutputs := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], OUTPUT_SIZE * SizeOf(Single), @Outputs[0]);
  BufferSamples := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], SAMPLE_SIZE * SizeOf(Single), @Samples[0]);
  BufferResults := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], RESULT_SIZE * SizeOf(Single), @Results[0]);

  FKernelDeltaOutput.Arguments[0].Access.SetBuffer(BufferOutputs);
  FKernelDeltaOutput.Arguments[1].Access.SetBuffer(BufferSamples);
  FKernelDeltaOutput.Arguments[2].Access.SetBuffer(BufferResults);
  FKernelDeltaOutput.Arguments[3].Access.SetValue<Cardinal>(OUTPUT_SIZE);

  FCommandQueue.EnqueueNDRangeKernel(FKernelDeltaOutput, TOCLGlobalDimensions.Create([OUTPUT_SIZE]));
  FCommandQueue.EnqueueReadBuffer(BufferResults, True, @Results[0]);

  for i := 0 to RESULT_SIZE - 1 do
    FLog.Add(Format('DeltaOutput[%d] = %.6f', [i, Results[i]]));

  FLog.Add('');
end;

procedure TNeuralNetworkOpenCL.FeedForward(ASample: PSample);
var
  i: Integer;
begin
  FLog.Add('FeedForward');

  for i := 0 to FTopology.Input - 1 do
  begin
    FNeuronsInput[i] := ASample^[i];
    FLog.AddFmt('FNeuronsInput[%d]=%.6f', [i, FNeuronsInput[i]]);
  end;
  FLog.Add('');
  FCommandQueue.EnqueueWriteBuffer(FBufferNeuronsInput, True, @FNeuronsInput[0]);

  {$REGION 'Calcular ativação INPUT --> HIDDEN'}
  FKernelMultiply.Arguments[0].Access.SetBuffer(FBufferNeuronsInput);
  FKernelMultiply.Arguments[1].Access.SetBuffer(FBufferWInputHidden);
  FKernelMultiply.Arguments[2].Access.SetBuffer(FBufferSumInputHidden);
  FKernelMultiply.Arguments[3].Access.SetValue<Cardinal>(FTopology.Input + 1); // +1 for BIAS
  FKernelMultiply.Arguments[4].Access.SetValue<Cardinal>(FTopology.Hidden);

  FCommandQueue.EnqueueNDRangeKernel(FKernelMultiply, TOCLGlobalDimensions.Create([FTopology.Input + 1, FTopology.Hidden]));
  FCommandQueue.EnqueueReadBuffer(FBufferSumInputHidden, True, @FSumInputHidden[0]);

//  for i := 0 to (FTopology.Input + 1) * FTopology.Hidden - 1 do
//    FLog.Add(Format('FSumInputHidden[%d] = %.6f', [i, FSumInputHidden[i]]));
//
//  FLog.Add('');

  FKernelSigmoide.Arguments[0].Access.SetBuffer(FBufferSumInputHidden);
  FKernelSigmoide.Arguments[1].Access.SetBuffer(FBufferNeuronsHidden);
  FKernelSigmoide.Arguments[2].Access.SetValue<Cardinal>(FTopology.Input + 1); // +1 for BIAS
  FKernelSigmoide.Arguments[3].Access.SetValue<Cardinal>(FTopology.Hidden);

  FCommandQueue.EnqueueNDRangeKernel(FKernelSigmoide, TOCLGlobalDimensions.Create([FTopology.Hidden]));
  FCommandQueue.EnqueueReadBuffer(FBufferNeuronsHidden, True, @FNeuronsHidden[0]);

  for i := 0 to FTopology.Hidden - 1 do
    FLog.Add(Format('FNeuronsHidden[%d] = %.6f', [i, FNeuronsHidden[i]]));

  FLog.Add('');
  {$ENDREGION}

  {$REGION 'Calcular ativação HIDDEN --> OUTPUT'}
  FKernelMultiply.Arguments[0].Access.SetBuffer(FBufferNeuronsHidden);
  FKernelMultiply.Arguments[1].Access.SetBuffer(FBufferWHiddenOutput);
  FKernelMultiply.Arguments[2].Access.SetBuffer(FBufferSumHiddenOutput);
  FKernelMultiply.Arguments[3].Access.SetValue<Cardinal>(FTopology.Hidden + 1); // +1 for BIAS
  FKernelMultiply.Arguments[4].Access.SetValue<Cardinal>(FTopology.Output);

  FCommandQueue.EnqueueNDRangeKernel(FKernelMultiply, TOCLGlobalDimensions.Create([FTopology.Hidden + 1, FTopology.Output]));
  FCommandQueue.EnqueueReadBuffer(FBufferSumHiddenOutput, True, @FSumHiddenOutput[0]);

//  for i := 0 to (FTopology.Hidden + 1) * FTopology.Output - 1 do
//    FLog.Add(Format('FSumHiddenOutput[%d] = %.6f', [i, FSumHiddenOutput[i]]));
//
//  FLog.Add('');

  FKernelSigmoide.Arguments[0].Access.SetBuffer(FBufferSumHiddenOutput);
  FKernelSigmoide.Arguments[1].Access.SetBuffer(FBufferNeuronsOutput);
  FKernelSigmoide.Arguments[2].Access.SetValue<Cardinal>(FTopology.Hidden + 1); // +1 for BIAS
  FKernelSigmoide.Arguments[3].Access.SetValue<Cardinal>(FTopology.Output);

  FCommandQueue.EnqueueNDRangeKernel(FKernelSigmoide, TOCLGlobalDimensions.Create([FTopology.Output]));
  FCommandQueue.EnqueueReadBuffer(FBufferNeuronsOutput, True, @FNeuronsOutput[0]);

  for i := 0 to FTopology.Output - 1 do
    FLog.Add(Format('FNeuronsOutput[%d] = %.6f', [i, FNeuronsOutput[i]]));

  FLog.Add('');
  {$ENDREGION}
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

procedure TNeuralNetworkOpenCL.Multiply;
const
  INPUT_SIZE = 3;
  OUTPUT_SIZE = 2;
  WEIGHT_SIZE = INPUT_SIZE * OUTPUT_SIZE;
  RESULT_SIZE = WEIGHT_SIZE;
var
  Inputs: array[0..INPUT_SIZE - 1] of Single;
  Outputs: array[0..OUTPUT_SIZE - 1] of Single;
  Weights: array[0..WEIGHT_SIZE - 1] of Single;
  Results: array[0..RESULT_SIZE - 1] of Single;

  BufferInputs: IOCLBuffer;
  BufferOutputs: IOCLBuffer;
  BufferWeights: IOCLBuffer;
  BufferResults: IOCLBuffer;

  i: Integer;
begin
  Inputs[0] := 2;
  Inputs[1] := 3;
  Inputs[2] := 5;

  Outputs[0] := 0;
  Outputs[1] := 0;

  for i := 0 to WEIGHT_SIZE - 1 do
  begin
    Weights[i] := i+1;
    Results[i] := 0;
  end;

  BufferInputs := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], INPUT_SIZE * SizeOf(Single), @Inputs[0]);
  BufferOutputs := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], OUTPUT_SIZE * SizeOf(Single), @Outputs[0]);
  BufferWeights := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], WEIGHT_SIZE * SizeOf(Single), @Weights[0]);
  BufferResults := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], RESULT_SIZE * SizeOf(Single), @Results[0]);

  FKernelMultiply.Arguments[0].Access.SetBuffer(BufferInputs);
  FKernelMultiply.Arguments[1].Access.SetBuffer(BufferWeights);
  FKernelMultiply.Arguments[2].Access.SetBuffer(BufferResults);
  FKernelMultiply.Arguments[3].Access.SetValue<Cardinal>(INPUT_SIZE);
  FKernelMultiply.Arguments[4].Access.SetValue<Cardinal>(OUTPUT_SIZE);

  FCommandQueue.EnqueueNDRangeKernel(FKernelMultiply, TOCLGlobalDimensions.Create([INPUT_SIZE, OUTPUT_SIZE]));
  FCommandQueue.EnqueueReadBuffer(BufferResults, True, @Results[0]);

  for i := 0 to RESULT_SIZE - 1 do
    FLog.Add(Format('Results[%d] = %f', [i, Results[i]]));

  FLog.Add('');

  FKernelSigmoide.Arguments[0].Access.SetBuffer(BufferResults);
  FKernelSigmoide.Arguments[1].Access.SetBuffer(BufferOutputs);
  FKernelSigmoide.Arguments[2].Access.SetValue<Cardinal>(INPUT_SIZE);
  FKernelSigmoide.Arguments[3].Access.SetValue<Cardinal>(OUTPUT_SIZE);

  FCommandQueue.EnqueueNDRangeKernel(FKernelSigmoide, TOCLGlobalDimensions.Create([OUTPUT_SIZE]));
  FCommandQueue.EnqueueReadBuffer(BufferOutputs, True, @Outputs[0]);

  for i := 0 to OUTPUT_SIZE - 1 do
    FLog.Add(Format('Outputs[%d] = %f', [i, Outputs[i]]));

  FLog.Add('');
end;

procedure TNeuralNetworkOpenCL.UpdateWeights;
const
  HIDDEN_SIZE = 3;
  OUTPUT_SIZE = 2;
  WEIGHT_SIZE = HIDDEN_SIZE * OUTPUT_SIZE;
  NEURON_SIZE = HIDDEN_SIZE;
  DELTA_SIZE  = OUTPUT_SIZE;
  ETA = 0.35;
var
  Weights: array[0..WEIGHT_SIZE - 1] of Single;
  Neurons: array[0..NEURON_SIZE - 1] of Single;
  Delta:   array[0..DELTA_SIZE - 1] of Single;

  BufferWeights: IOCLBuffer;
  BufferNeurons: IOCLBuffer;
  BufferDelta: IOCLBuffer;

  i: Integer;
begin
  Weights[0] := 1;
  Weights[1] := 2;
  Weights[2] := 3;
  Weights[3] := 4;
  Weights[4] := 5;
  Weights[5] := 6;

  Neurons[0] := 0.2;
  Neurons[1] := 0.3;
  Neurons[2] := 0.4;

  Delta[0] := 0.04192;
  Delta[1] := 0.10332;

  BufferWeights := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], WEIGHT_SIZE * SizeOf(Single), @Weights[0]);
  BufferNeurons := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], NEURON_SIZE * SizeOf(Single), @Neurons[0]);
  BufferDelta := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], DELTA_SIZE * SizeOf(Single), @Delta[0]);

  FKernelUpdateWeights.Arguments[0].Access.SetBuffer(BufferWeights);
  FKernelUpdateWeights.Arguments[1].Access.SetBuffer(BufferNeurons);
  FKernelUpdateWeights.Arguments[2].Access.SetBuffer(BufferDelta);
  FKernelUpdateWeights.Arguments[3].Access.SetValue<Cardinal>(NEURON_SIZE);
  FKernelUpdateWeights.Arguments[4].Access.SetValue<Cardinal>(DELTA_SIZE);
  FKernelUpdateWeights.Arguments[5].Access.SetValue<Single>(ETA);

  FCommandQueue.EnqueueNDRangeKernel(FKernelUpdateWeights, TOCLGlobalDimensions.Create([NEURON_SIZE, DELTA_SIZE]));
  FCommandQueue.EnqueueReadBuffer(BufferWeights, True, @Weights[0]);

  for i := 0 to WEIGHT_SIZE - 1 do
    FLog.Add(Format('Weights[%d] = %.6f', [i, Weights[i]]));

  FLog.Add('');
end;

procedure TNeuralNetworkOpenCL.BackPropagation(ASample: PSample);
var
  i, iSample: Integer;
begin
  FLog.Add('BackPropagation');
  for i := 0 to FTopology.Output - 1 do
  begin
    iSample := FTopology.Input + i;
    FSampleOutput[i] := ASample^[iSample];

    FLog.AddFmt('FSampleOutput[%d]=%.6f', [i, FSampleOutput[i]]);
  end;
  FLog.Add('');
  FCommandQueue.EnqueueWriteBuffer(FBufferSampleOutput, True, @FSampleOutput[0]);

  {$REGION 'Delta OUTPUT'}
  FKernelDeltaOutput.Arguments[0].Access.SetBuffer(FBufferNeuronsOutput);
  FKernelDeltaOutput.Arguments[1].Access.SetBuffer(FBufferSampleOutput);
  FKernelDeltaOutput.Arguments[2].Access.SetBuffer(FBufferDeltaOutput);
  FKernelDeltaOutput.Arguments[3].Access.SetValue<Cardinal>(FTopology.Output);

  FCommandQueue.EnqueueNDRangeKernel(FKernelDeltaOutput, TOCLGlobalDimensions.Create([FTopology.Output]));
  FCommandQueue.EnqueueReadBuffer(FBufferDeltaOutput, True, @FDeltaOutput[0]);

  for i := 0 to FTopology.Output - 1 do
    FLog.Add(Format('DeltaOutput[%d] = %.6f', [i, FDeltaOutput[i]]));

  FLog.Add('');
  {$ENDREGION}

  {$REGION 'Delta HIDDEN'}
  FKernelDeltaHidden.Arguments[0].Access.SetBuffer(FBufferNeuronsHidden);
  FKernelDeltaHidden.Arguments[1].Access.SetBuffer(FBufferDeltaOutput);
  FKernelDeltaHidden.Arguments[2].Access.SetBuffer(FBufferWHiddenOutput);
  FKernelDeltaHidden.Arguments[3].Access.SetBuffer(FBufferDeltaHidden);
  FKernelDeltaHidden.Arguments[4].Access.SetValue<Cardinal>(FTopology.Hidden + 1); // +1 for BIAS
  FKernelDeltaHidden.Arguments[5].Access.SetValue<Cardinal>(FTopology.Output);

  FCommandQueue.EnqueueNDRangeKernel(FKernelDeltaHidden, TOCLGlobalDimensions.Create([FTopology.Hidden + 1]));  // +1 for BIAS
  FCommandQueue.EnqueueReadBuffer(FBufferDeltaHidden, True, @FDeltaHidden[0]);

  for i := 0 to FTopology.Hidden {+1 for BIAS} do
    FLog.Add(Format('DeltaHidden[%d] = %.6f', [i, FDeltaHidden[i]]));

  FLog.Add('');
  {$ENDREGION}


  {$REGION 'Update Weights HIDDEN --> OUTPUT'}
  FKernelUpdateWeights.Arguments[0].Access.SetBuffer(FBufferWHiddenOutput);
  FKernelUpdateWeights.Arguments[1].Access.SetBuffer(FBufferNeuronsHidden);
  FKernelUpdateWeights.Arguments[2].Access.SetBuffer(FBufferDeltaOutput);
  FKernelUpdateWeights.Arguments[3].Access.SetValue<Cardinal>(FTopology.Hidden + 1); // +1 for BIAS
  FKernelUpdateWeights.Arguments[4].Access.SetValue<Cardinal>(FTopology.Output);
  FKernelUpdateWeights.Arguments[5].Access.SetValue<Single>(ETA);

  FCommandQueue.EnqueueNDRangeKernel(FKernelUpdateWeights, TOCLGlobalDimensions.Create([FTopology.Hidden + 1, FTopology.Output]));
  FCommandQueue.EnqueueReadBuffer(FBufferWHiddenOutput, True, @FWHiddenOutput[0]);

  for i := 0 to ((FTopology.Hidden + 1) * FTopology.Output) - 1 do
    FLog.Add(Format('FWHiddenOutput[%d] = %.6f', [i, FWHiddenOutput[i]]));

  FLog.Add('');
  {$ENDREGION}

  {$REGION 'Update Weights INPUT --> HIDDEN'}
  FKernelUpdateWeights.Arguments[0].Access.SetBuffer(FBufferWInputHidden);
  FKernelUpdateWeights.Arguments[1].Access.SetBuffer(FBufferNeuronsInput);
  FKernelUpdateWeights.Arguments[2].Access.SetBuffer(FBufferDeltaHidden);
  FKernelUpdateWeights.Arguments[3].Access.SetValue<Cardinal>(FTopology.Input + 1); // +1 for BIAS
  FKernelUpdateWeights.Arguments[4].Access.SetValue<Cardinal>(FTopology.Hidden); // +1 ???
  FKernelUpdateWeights.Arguments[5].Access.SetValue<Single>(ETA);

  FCommandQueue.EnqueueNDRangeKernel(FKernelUpdateWeights, TOCLGlobalDimensions.Create([FTopology.Input + 1, FTopology.Hidden]));
  FCommandQueue.EnqueueReadBuffer(FBufferWInputHidden, True, @FWInputHidden[0]);

  for i := 0 to ((FTopology.Input + 1) * FTopology.Hidden) - 1 do
    FLog.Add(Format('FWInputHidden[%d] = %.6f', [i, FWInputHidden[i]]));

  FLog.Add('');
  {$ENDREGION}
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

  FCommandQueue := FContext.CreateCommandQueue();
end;

constructor TNeuralNetworkOpenCL.Create(ATopology: TTopology);
begin
  inherited;

  SetLength(FSumInputHidden, (FTopology.Input + 1) * FTopology.Hidden);
  SetLength(FSumHiddenOutput, (FTopology.Hidden + 1) * FTopology.Output);
  SetLength(FSampleOutput, FTopology.Output);

  BuildKernel;

  FBufferNeuronsInput := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (ATopology.Input + 1) * SizeOf(Single), @FNeuronsInput[0]); // +1 for BIAS
  FBufferNeuronsHidden := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (ATopology.Hidden + 1) * SizeOf(Single), @FNeuronsHidden[0]); // +1 for BIAS
  FBufferNeuronsOutput := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], ATopology.Output * SizeOf(Single), @FNeuronsOutput[0]);

  FBufferWInputHidden := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (FTopology.Input + 1) * FTopology.Hidden * SizeOf(Single), @FWInputHidden[0]);
  FBufferWHiddenOutput := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (FTopology.Hidden + 1) * FTopology.Output * SizeOf(Single), @FWHiddenOutput[0]);

  FBufferSumInputHidden := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (FTopology.Input + 1) * FTopology.Hidden * SizeOf(Single), @FSumInputHidden[0]);
  FBufferSumHiddenOutput := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (FTopology.Hidden + 1) * FTopology.Output * SizeOf(Single), @FSumHiddenOutput[0]);

  FBufferDeltaOutput := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (ATopology.Output) * SizeOf(Single), @FDeltaOutput[0]);
  FBufferDeltaHidden := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (ATopology.Hidden + 1) * SizeOf(Single), @FDeltaHidden[0]); // +1 for BIAS

  FBufferSampleOutput := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], ATopology.Output * SizeOf(Single), @FSampleOutput[0]);
end;

end.
