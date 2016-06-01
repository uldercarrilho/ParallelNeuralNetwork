unit uNeuralNetworkOpenCLTests;

interface

uses
  uNeuralNetworkOpenCL;

type
  TNeuralNetworkOpenCLTests = class(TNeuralNetworkOpenCL)
  public
    procedure Multiply;
    procedure DeltaOutput;
    procedure DeltaHidden;
    procedure UpdateWeights;
    procedure Params;
  end;

implementation

uses
  Mitov.OpenCL, uHelpers;

{ TNeuralNetworkOpenCLTests }

procedure TNeuralNetworkOpenCLTests.DeltaHidden;
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
    FLog.AddFmt('DeltaHidden[%d] = %.6f', [i, Results[i]]);
  FLog.Add('');
end;

procedure TNeuralNetworkOpenCLTests.DeltaOutput;
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
    FLog.AddFmt('DeltaOutput[%d] = %.6f', [i, Results[i]]);
  FLog.Add('');
end;

procedure TNeuralNetworkOpenCLTests.Multiply;
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
    FLog.AddFmt('Results[%d] = %f', [i, Results[i]]);
  FLog.Add('');

  FKernelSigmoide.Arguments[0].Access.SetBuffer(BufferResults);
  FKernelSigmoide.Arguments[1].Access.SetBuffer(BufferOutputs);
  FKernelSigmoide.Arguments[2].Access.SetValue<Cardinal>(INPUT_SIZE);
  FKernelSigmoide.Arguments[3].Access.SetValue<Cardinal>(OUTPUT_SIZE);

  FCommandQueue.EnqueueNDRangeKernel(FKernelSigmoide, TOCLGlobalDimensions.Create([OUTPUT_SIZE]));
  FCommandQueue.EnqueueReadBuffer(BufferOutputs, True, @Outputs[0]);

  for i := 0 to OUTPUT_SIZE - 1 do
    FLog.AddFmt('Outputs[%d] = %f', [i, Outputs[i]]);
  FLog.Add('');
end;

procedure TNeuralNetworkOpenCLTests.Params;
const
  VECTOR_SIZE = 1000;
var
  Values: array[0..VECTOR_SIZE] of Single;
  Result: array[0..VECTOR_SIZE] of Single;
  BufferValues: IOCLBuffer;
  BufferResult: IOCLBuffer;
  i: Integer;
begin
  for i := 0 to VECTOR_SIZE - 1 do
  begin
    Values[i] := i;
    Result[i] := 0;
  end;

  BufferValues := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], VECTOR_SIZE * SizeOf(Single), @Values[0]);
  BufferResult := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], VECTOR_SIZE * SizeOf(Single), @Result[0]);

  FKernelParams.Arguments[0].Access.SetBuffer(BufferResult);
  FKernelParams.Arguments[1].Access.SetBuffer(BufferValues);
  FKernelParams.Arguments[2].Access.SetBuffer(BufferValues);
  FKernelParams.Arguments[3].Access.SetBuffer(BufferValues);
  FKernelParams.Arguments[4].Access.SetBuffer(BufferValues);
  FKernelParams.Arguments[5].Access.SetBuffer(BufferValues);
  FKernelParams.Arguments[6].Access.SetBuffer(BufferValues);
  FKernelParams.Arguments[7].Access.SetBuffer(BufferValues);
  FKernelParams.Arguments[8].Access.SetBuffer(BufferValues);
  FKernelParams.Arguments[9].Access.SetBuffer(BufferValues);

  FCommandQueue.EnqueueNDRangeKernel(FKernelParams, TOCLGlobalDimensions.Create([VECTOR_SIZE]));
  FCommandQueue.EnqueueReadBuffer(BufferResult, True, @Result[0]);

  for i := 0 to 10 do
    FLog.AddFmt('Result[%d] = %.6f', [i, Result[i]]);
  FLog.Add('');
end;

procedure TNeuralNetworkOpenCLTests.UpdateWeights;
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
    FLog.AddFmt('Weights[%d] = %.6f', [i, Weights[i]]);
  FLog.Add('');
end;

end.
