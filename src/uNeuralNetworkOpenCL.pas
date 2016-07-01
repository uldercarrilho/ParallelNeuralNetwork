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
    FKernelGemm1: IOCLKernel;
    FKernelActive: IOCLKernel;

    FBufferSamples: IOCLBuffer;
    // o FBufferNeuronsInput foi substituído pelo FBufferSamples para evitar cópia desnecessária de dados
    //FBufferNeuronsInput: IOCLBuffer;
    FBufferNeuronsHidden: IOCLBuffer;
    FBufferNeuronsOutput: IOCLBuffer;

    FBufferWInputHidden: IOCLBuffer;
    FBufferWHiddenOutput: IOCLBuffer;

    FBufferSumInputHidden: IOCLBuffer;
    FBufferSumHiddenOutput: IOCLBuffer;

    FBufferDeltaOutput: IOCLBuffer;
    FBufferDeltaHidden: IOCLBuffer;

    FSumInputHidden: TVector1D;
    FSumHiddenOutput: TVector1D;

    /// <summary>
    ///  Obtém o código fonte do Kernel OpenCL vinculado como recurso ao executável.
    /// </summary>
    /// <param name="AResourceName">
    ///  Nome do recurso vinculado ao executável.
    /// </param>
    /// <returns>
    ///  Texto do código fonte.
    /// </returns>
    /// <remarks>
    ///  Método utilizado para compilar o kernel OpenCL.
    /// </remarks>
    function GetOpenCLSource(const AResourceName: string): string;
    /// <summary>
    ///  Selecione o device (GPU), carrega o código fonte e compila os kernels OpenCL. Este método prepara para a
    ///  execução dos kernels na GPU.
    /// </summary>
    procedure BuildKernel;
    /// <summary>
    ///  Cria os objetos que representam os dados que serão transferidos para a GPU.
    /// </summary>
    procedure CreateBuffers;
    /// <summary>
    ///  Envia os dados do HOST para a GPU.
    /// </summary>
    procedure WriteBufferToGPU;
    /// <summary>
    ///  Calcula a etapa de FeedForward do algoritmo de aprendizagem da rede neural. O cálculo é realizado para todas
    ///  as camadas da rede neural, ou seja, Input -> Hidden e Hidden -> Output.
    /// </summary>
    /// <param name="iSample">
    ///  Índice da amostra que está sendo computada.
    /// </param>
    procedure FeedForward(iSample: Cardinal); override;
    procedure FeedForwardOneKernel(iSample: Cardinal);
    /// <summary>
    ///  Calcula a etapa de backpropagation do algoritmo de aprendizagem da rede neural. Nesta etapa, é calculado o
    ///  Delta que representa o quanto a resposta está diferente do esperado e depois utiliza este valor para atualizar
    ///  os pesos entre os neurônios, iniciando na camada de saída até a camada de entrada.
    /// </summary>
    /// <param name="iSample">
    ///  Índice da amostra que está sendo computada.
    /// </param>
    procedure BackPropagation(iSample: Cardinal); override;
  public
    constructor Create(ATopology: TTopology); override;
    destructor Destroy; override;
    /// <summary>
    ///  Executa o método BuildKernel.
    /// </summary>
    procedure Prepare; override;
    /// <summary>
    ///  Executa a etapa de aprendizagem, ou seja, computa o FeedForward e Backpropagation para todas as entradas do
    ///  conjunto de amostras, repetindo AEpochs vezes.
    /// </summary>
    /// <remarks>
    ///  A condição de parada do método é computar todas as entradas do conjunto de amostras. Não há um controle de
    ///  parada com base na margem de erro do previsto e computado.
    /// </remarks>
    procedure Learn(AEpochs: Cardinal); override;
    procedure Multiply;
    procedure LogMatrix(Matrix: PVector1D; M, N: Integer);
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
end;

destructor TNeuralNetworkOpenCL.Destroy;
begin
  SetLength(FSumInputHidden, 0);
  SetLength(FSumHiddenOutput, 0);

  inherited;
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

procedure TNeuralNetworkOpenCL.BuildKernel;
var
  sOpenCLSource: string;
  APlatform: IOCLPlatform;
  AProgram: IOCLProgram;
  i: Integer;
  TickCount: Cardinal;
begin
  FLog.Add('Building kernel OpenCL');
  TickCount := TThread.GetTickCount;

  APlatform := TOpenCL.Platforms[0];

  // retira da lista todos os devices que não são GPU
  i := APlatform.Devices.Count - 1;
  while i >= 0 do
  begin
    if APlatform.Devices.Items[i].DeviceTypes <> [TOCLDeviceType.GPU] then
      APlatform.Devices.Delete(i);
    Dec(i);
  end;
  // cria o contexto apenas para os devices GPUs
  FContext := TOCLContext.Create(APlatform.Devices);

  // carrega o código fonte do kernel OpenCL e compila para os DEVICES
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

  // obtém a referência para os kernels
  FKernelMultiply := AProgram.Kernel['multiply'];
  FKernelSigmoide := AProgram.Kernel['sigmoide'];
  FKernelDeltaOutput := AProgram.Kernel['calculateDeltaOutput'];
  FKernelDeltaHidden := AProgram.Kernel['calculateDeltaHidden'];
  FKernelUpdateWeights := AProgram.Kernel['updateWeights'];
  FKernelParams := AProgram.Kernel['params'];
  FKernelGemm1 := AProgram.Kernel['myGEMM1'];
  FKernelActive := AProgram.Kernel['active'];

  // cria a fila de execução dos kernels e demais comandos
  FCommandQueue := FContext.CreateCommandQueue();

  TickCount := TThread.GetTickCount - TickCount;
  FLog.AddFmt('Elapsed time: %d ms', [TickCount]);
end;

procedure TNeuralNetworkOpenCL.Learn(AEpochs: Cardinal);
var
  i, j: Integer;
begin
  CreateBuffers;
  WriteBufferToGPU;

  for i := 1 to AEpochs do
  begin
    FLog.AddFmt('Training epoch %d', [i]);
    for j := 0 to FSamplesSet.SamplesCount - 1 do
    begin
      FeedForward(j);
      FCommandQueue.Finish;
      BackPropagation(j);
      FCommandQueue.Finish;
    end;
  end;
end;

procedure TNeuralNetworkOpenCL.LogMatrix(Matrix: PVector1D; M, N: Integer);
var
  i, j: Integer;
  Line: string;
begin
  for i := 0 to M - 1 do
  begin
    Line := '';
    for j := 0 to N - 1 do
      Line := Line + FloatToStr(Matrix^[i*N + j]) + ' ';

    FLog.Add(Line);
  end;
end;

procedure TNeuralNetworkOpenCL.Multiply;
const
  M = 1;
  N = 3;
  K = 4;
var
  MatrixA, MatrixB, MatrixC: TVector1D;
  BufferA, BufferB, BufferC: IOCLBuffer;
begin
  SetLength(MatrixA, M * K);
  SetLength(MatrixB, K * N);
  SetLength(MatrixC, M * N);

  MatrixA[0] := 1;
  MatrixA[1] := 2;
  MatrixA[2] := 3;
  MatrixA[3] := 4;

  MatrixB[0]  := 1;
  MatrixB[1]  := 2;
  MatrixB[2]  := 3;
  MatrixB[3]  := 4;
  MatrixB[4]  := 5;
  MatrixB[5]  := 6;
  MatrixB[6]  := 7;
  MatrixB[7]  := 8;
  MatrixB[8]  := 9;
  MatrixB[9]  := 10;
  MatrixB[10] := 11;
  MatrixB[11] := 12;

  MatrixC[0] := 0;
  MatrixC[1] := 0;
  MatrixC[2] := 0;

  FLog.Add('Matrix A');
  LogMatrix(@MatrixA, M, K);
  FLog.Add('');

  FLog.Add('Matrix B');
  LogMatrix(@MatrixB, K, N);
  FLog.Add('');

  FLog.Add('Matrix C');
  LogMatrix(@MatrixC, M, N);
  FLog.Add('');

  BufferA := FContext.CreateBuffer([TOCLMemoryFlag.ReadOnly, TOCLMemoryFlag.UseHostPtr], M * K * SizeOf(Single), @MatrixA[0]);
  BufferB := FContext.CreateBuffer([TOCLMemoryFlag.ReadOnly, TOCLMemoryFlag.UseHostPtr], K * N * SizeOf(Single), @MatrixB[0]);
  BufferC := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], M * N * SizeOf(Single), @MatrixC[0]);

  FCommandQueue.EnqueueWriteBuffer(BufferA, True, @MatrixA[0]);
  FCommandQueue.EnqueueWriteBuffer(BufferB, True, @MatrixB[0]);
  FCommandQueue.EnqueueWriteBuffer(BufferC, True, @MatrixC[0]);

  FKernelGemm1.Arguments[0].Access.SetValue<Integer>(M);
  FKernelGemm1.Arguments[1].Access.SetValue<Integer>(N);
  FKernelGemm1.Arguments[2].Access.SetValue<Integer>(K);
  FKernelGemm1.Arguments[3].Access.SetBuffer(BufferA);
  FKernelGemm1.Arguments[4].Access.SetBuffer(BufferB);
  FKernelGemm1.Arguments[5].Access.SetBuffer(BufferC);

  FCommandQueue.EnqueueNDRangeKernel(FKernelGemm1, TOCLGlobalDimensions.Create([M, N]));

  FCommandQueue.EnqueueReadBuffer(BufferC, True, @MatrixC[0]);

  FLog.Add('Matrix C = A * B');
  LogMatrix(@MatrixC, M, N);
  FLog.Add('');

  SetLength(MatrixA, 0);
  SetLength(MatrixB, 0);
  SetLength(MatrixC, 0);
end;

procedure TNeuralNetworkOpenCL.Prepare;
begin
  inherited;
  BuildKernel;
end;

procedure TNeuralNetworkOpenCL.CreateBuffers;
var
  SampleSize: Word;
  BufferSize: Cardinal;
  TickCount: Cardinal;
begin
  FLog.Add('Creating buffers');
  TickCount := TThread.GetTickCount;

  SampleSize := FTopology.Input + 1 + FTopology.Output; // +1 for BIAS
  BufferSize := SampleSize * FSamplesSet.SamplesCount * SizeOf(Single);

  FBufferSamples := FContext.CreateBuffer([TOCLMemoryFlag.ReadOnly, TOCLMemoryFlag.UseHostPtr], BufferSize, @FSamplesSet.Samples1D[0]);
  // o FBufferNeuronsInput foi substituído pelo FBufferSamples para evitar cópia desnecessária de dados
  //FBufferNeuronsInput := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (FTopology.Input + 1) * SizeOf(Single), @FNeuronsInput[0]); // +1 for BIAS
  FBufferNeuronsHidden := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (FTopology.Hidden + 1) * SizeOf(Single), @FNeuronsHidden[0]); // +1 for BIAS
  FBufferNeuronsOutput := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], FTopology.Output * SizeOf(Single), @FNeuronsOutput[0]);

  FBufferWInputHidden := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (FTopology.Input + 1) * FTopology.Hidden * SizeOf(Single), @FWeights1DInputHidden[0]);
  FBufferWHiddenOutput := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (FTopology.Hidden + 1) * FTopology.Output * SizeOf(Single), @FWeights1DHiddenOutput[0]);

  FBufferSumInputHidden := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (FTopology.Input + 1) * FTopology.Hidden * SizeOf(Single), @FSumInputHidden[0]);
  FBufferSumHiddenOutput := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (FTopology.Hidden + 1) * FTopology.Output * SizeOf(Single), @FSumHiddenOutput[0]);

  FBufferDeltaOutput := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (FTopology.Output) * SizeOf(Single), @FDeltaOutput[0]);
  FBufferDeltaHidden := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (FTopology.Hidden + 1) * SizeOf(Single), @FDeltaHidden[0]); // +1 for BIAS

  TickCount := TThread.GetTickCount - TickCount;
  FLog.AddFmt('Elapsed time: %d ms', [TickCount]);
end;

procedure TNeuralNetworkOpenCL.WriteBufferToGPU;
var
  TickCount: Cardinal;
begin
  FLog.Add('Sending buffers to GPU');
  TickCount := TThread.GetTickCount;

  FCommandQueue.EnqueueWriteBuffer(FBufferSamples, True, @FSamplesSet.Samples1D[0]);
  // o FBufferNeuronsInput foi substituído pelo FBufferSamples para evitar cópia desnecessária de dados
  //FCommandQueue.EnqueueWriteBuffer(FBufferNeuronsInput, True, @FNeuronsInput[0]);
  FCommandQueue.EnqueueWriteBuffer(FBufferNeuronsHidden, True, @FNeuronsHidden[0]);
  FCommandQueue.EnqueueWriteBuffer(FBufferNeuronsOutput, True, @FNeuronsOutput[0]);

  FCommandQueue.EnqueueWriteBuffer(FBufferWInputHidden, True, @FWeights1DInputHidden[0]);
  FCommandQueue.EnqueueWriteBuffer(FBufferWHiddenOutput, True, @FWeights1DHiddenOutput[0]);

  FCommandQueue.EnqueueWriteBuffer(FBufferSumInputHidden, True, @FSumInputHidden[0]);
  FCommandQueue.EnqueueWriteBuffer(FBufferSumHiddenOutput, True, @FSumHiddenOutput[0]);

  FCommandQueue.EnqueueWriteBuffer(FBufferDeltaOutput, True, @FDeltaOutput[0]);
  FCommandQueue.EnqueueWriteBuffer(FBufferDeltaHidden, True, @FDeltaHidden[0]);

  FCommandQueue.Flush;
  FCommandQueue.Finish;

  TickCount := TThread.GetTickCount - TickCount;
  FLog.AddFmt('Elapsed time: %d ms', [TickCount]);
end;

procedure TNeuralNetworkOpenCL.FeedForward(iSample: Cardinal);
var
  InputOffSet: Cardinal;
begin
  FeedForwardOneKernel(iSample);
  Exit;

  {$REGION 'Calcular ativação INPUT --> HIDDEN'}
  InputOffSet := iSample * (FTopology.Input + 1 + FTopology.Output); // +1 for BIAS

  FKernelMultiply.Arguments[0].Access.SetBuffer(FBufferSamples);
  FKernelMultiply.Arguments[1].Access.SetBuffer(FBufferWInputHidden);
  FKernelMultiply.Arguments[2].Access.SetBuffer(FBufferSumInputHidden);
  FKernelMultiply.Arguments[3].Access.SetValue<Cardinal>(InputOffSet);
  FKernelMultiply.Arguments[4].Access.SetValue<Cardinal>(FTopology.Input + 1); // +1 for BIAS
  FKernelMultiply.Arguments[5].Access.SetValue<Cardinal>(FTopology.Hidden);

  FCommandQueue.EnqueueNDRangeKernel(FKernelMultiply, TOCLGlobalDimensions.Create([FTopology.Input + 1, FTopology.Hidden]));  // +1 for BIAS
  // DEBUG
  // FCommandQueue.Finish;
  // FCommandQueue.EnqueueReadBuffer(FBufferSumInputHidden, True, @FSumInputHidden[0]);

  // for i := 0 to (FTopology.Input + 1) * FTopology.Hidden - 1 do
  //   FLog.Add(Format('FSumInputHidden[%d] = %.6f', [i, FSumInputHidden[i]]));
  // FLog.Add('');

  FKernelSigmoide.Arguments[0].Access.SetBuffer(FBufferSumInputHidden);
  FKernelSigmoide.Arguments[1].Access.SetBuffer(FBufferNeuronsHidden);
  FKernelSigmoide.Arguments[2].Access.SetValue<Cardinal>(FTopology.Input + 1); // +1 for BIAS
  FKernelSigmoide.Arguments[3].Access.SetValue<Cardinal>(FTopology.Hidden);

  FCommandQueue.EnqueueNDRangeKernel(FKernelSigmoide, TOCLGlobalDimensions.Create([FTopology.Hidden]));
  // DEBUG
  // FCommandQueue.Finish;
  // FCommandQueue.EnqueueReadBuffer(FBufferNeuronsHidden, True, @FNeuronsHidden[0]);

  // for i := 0 to FTopology.Hidden - 1 do
  //   FLog.Add(Format('FNeuronsHidden[%d] = %.6f', [i, FNeuronsHidden[i]]));
  // FLog.Add('');
  {$ENDREGION}

  {$REGION 'Calcular ativação HIDDEN --> OUTPUT'}
  FKernelMultiply.Arguments[0].Access.SetBuffer(FBufferNeuronsHidden);
  FKernelMultiply.Arguments[1].Access.SetBuffer(FBufferWHiddenOutput);
  FKernelMultiply.Arguments[2].Access.SetBuffer(FBufferSumHiddenOutput);
  FKernelMultiply.Arguments[3].Access.SetValue<Cardinal>(0);
  FKernelMultiply.Arguments[4].Access.SetValue<Cardinal>(FTopology.Hidden + 1); // +1 for BIAS
  FKernelMultiply.Arguments[5].Access.SetValue<Cardinal>(FTopology.Output);

  //TOCLGlobalAndLocalDimensions.Create(10, 2);
  FCommandQueue.EnqueueNDRangeKernel(FKernelMultiply, TOCLGlobalDimensions.Create([FTopology.Hidden + 1, FTopology.Output]));
  // DEBUG
  // FCommandQueue.Finish;
  // FCommandQueue.EnqueueReadBuffer(FBufferSumHiddenOutput, True, @FSumHiddenOutput[0]);

  // for i := 0 to (FTopology.Hidden + 1) * FTopology.Output - 1 do
  //   FLog.Add(Format('FSumHiddenOutput[%d] = %.6f', [i, FSumHiddenOutput[i]]));
  // FLog.Add('');

  FKernelSigmoide.Arguments[0].Access.SetBuffer(FBufferSumHiddenOutput);
  FKernelSigmoide.Arguments[1].Access.SetBuffer(FBufferNeuronsOutput);
  FKernelSigmoide.Arguments[2].Access.SetValue<Cardinal>(FTopology.Hidden + 1); // +1 for BIAS
  FKernelSigmoide.Arguments[3].Access.SetValue<Cardinal>(FTopology.Output);

  FCommandQueue.EnqueueNDRangeKernel(FKernelSigmoide, TOCLGlobalDimensions.Create([FTopology.Output]));
  // DEBUG
  // FCommandQueue.Finish;
  // FCommandQueue.EnqueueReadBuffer(FBufferNeuronsOutput, True, @FNeuronsOutput[0]);

  // for i := 0 to FTopology.Output - 1 do
  //   FLog.Add(Format('FNeuronsOutput[%d] = %.6f', [i, FNeuronsOutput[i]]));
  // FLog.Add('');
  {$ENDREGION}
end;

procedure TNeuralNetworkOpenCL.FeedForwardOneKernel(iSample: Cardinal);
var
  M, N, K: Integer;
  InputOffSet: Integer;
begin
// 	Média: 1153 -- FeedForward
//  Média: 921  -- FeedForwardOneKernel

  {$REGION 'Calcular ativação INPUT --> HIDDEN'}
  M := 1; // 1 amostra
  N := FTopology.Hidden;
  K := FTopology.Input + 1; // +1 for BIAS
  InputOffSet := iSample * (FTopology.Input + 1 + FTopology.Output); // +1 for BIAS

  FKernelActive.Arguments[0].Access.SetValue<Integer>(M);
  FKernelActive.Arguments[1].Access.SetValue<Integer>(N);
  FKernelActive.Arguments[2].Access.SetValue<Integer>(K);
  FKernelActive.Arguments[3].Access.SetValue<Integer>(InputOffSet);
  FKernelActive.Arguments[4].Access.SetBuffer(FBufferSamples);
  FKernelActive.Arguments[5].Access.SetBuffer(FBufferWInputHidden);
  FKernelActive.Arguments[6].Access.SetBuffer(FBufferNeuronsHidden);

  FCommandQueue.EnqueueNDRangeKernel(FKernelActive, TOCLGlobalDimensions.Create([M, N]));
  {$ENDREGION}

  {$REGION 'Calcular ativação HIDDEN --> OUTPUT'}
  M := 1; // 1 amostra
  N := FTopology.Output;
  K := FTopology.Hidden + 1; // +1 for BIAS

  FKernelActive.Arguments[0].Access.SetValue<Integer>(M);
  FKernelActive.Arguments[1].Access.SetValue<Integer>(N);
  FKernelActive.Arguments[2].Access.SetValue<Integer>(K);
  FKernelActive.Arguments[3].Access.SetValue<Integer>(0);
  FKernelActive.Arguments[4].Access.SetBuffer(FBufferNeuronsHidden);
  FKernelActive.Arguments[5].Access.SetBuffer(FBufferWHiddenOutput);
  FKernelActive.Arguments[6].Access.SetBuffer(FBufferNeuronsOutput);

  FCommandQueue.EnqueueNDRangeKernel(FKernelActive, TOCLGlobalDimensions.Create([M, N]));
  {$ENDREGION}
end;

procedure TNeuralNetworkOpenCL.BackPropagation(iSample: Cardinal);
var
  //i, iOutput: Integer;
  OutputOffset: Cardinal;
  NeuronOffset: Cardinal;
begin
  {$REGION 'Delta OUTPUT'}
  OutputOffset := iSample * (FTopology.Input + 1 + FTopology.Output) + FTopology.Input; // +1 for BIAS

  FKernelDeltaOutput.Arguments[0].Access.SetBuffer(FBufferNeuronsOutput);
  FKernelDeltaOutput.Arguments[1].Access.SetBuffer(FBufferSamples);
  FKernelDeltaOutput.Arguments[2].Access.SetBuffer(FBufferDeltaOutput);
  FKernelDeltaOutput.Arguments[3].Access.SetValue<Cardinal>(FTopology.Output);
  FKernelDeltaOutput.Arguments[4].Access.SetValue<Cardinal>(OutputOffset);

  FCommandQueue.EnqueueNDRangeKernel(FKernelDeltaOutput, TOCLGlobalDimensions.Create([FTopology.Output]));
  // DEBUG
  // FCommandQueue.Finish;
  // FCommandQueue.EnqueueReadBuffer(FBufferDeltaOutput, True, @FDeltaOutput[0]);

  // for i := 0 to FTopology.Output - 1 do
  //   FLog.Add(Format('DeltaOutput[%d] = %.6f', [i, FDeltaOutput[i]]));
  // FLog.Add('');
  {$ENDREGION}

  {$REGION 'Delta HIDDEN'}
  FKernelDeltaHidden.Arguments[0].Access.SetBuffer(FBufferNeuronsHidden);
  FKernelDeltaHidden.Arguments[1].Access.SetBuffer(FBufferDeltaOutput);
  FKernelDeltaHidden.Arguments[2].Access.SetBuffer(FBufferWHiddenOutput);
  FKernelDeltaHidden.Arguments[3].Access.SetBuffer(FBufferDeltaHidden);
  FKernelDeltaHidden.Arguments[4].Access.SetValue<Cardinal>(FTopology.Hidden + 1); // +1 for BIAS
  FKernelDeltaHidden.Arguments[5].Access.SetValue<Cardinal>(FTopology.Output);

  FCommandQueue.EnqueueNDRangeKernel(FKernelDeltaHidden, TOCLGlobalDimensions.Create([FTopology.Hidden + 1]));  // +1 for BIAS
  // DEBUG
  // FCommandQueue.Finish;
  // FCommandQueue.EnqueueReadBuffer(FBufferDeltaHidden, True, @FDeltaHidden[0]);

  // for i := 0 to FTopology.Hidden {+1 for BIAS} do
  //   FLog.Add(Format('DeltaHidden[%d] = %.6f', [i, FDeltaHidden[i]]));
  // FLog.Add('');
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
  // DEBUG
  // FCommandQueue.Finish;
  // FCommandQueue.EnqueueReadBuffer(FBufferWHiddenOutput, True, @FWHiddenOutput[0]);

  // for i := 0 to ((FTopology.Hidden + 1) * FTopology.Output) - 1 do
  //   FLog.Add(Format('FWHiddenOutput[%d] = %.6f', [i, FWHiddenOutput[i]]));
  // FLog.Add('');
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
  // DEBUG
  // FCommandQueue.Finish;
  // FCommandQueue.EnqueueReadBuffer(FBufferWInputHidden, True, @FWInputHidden[0]);

  // for i := 0 to ((FTopology.Input + 1) * FTopology.Hidden) - 1 do
  //   FLog.Add(Format('FWInputHidden[%d] = %.6f', [i, FWInputHidden[i]]));
  // FLog.Add('');
  {$ENDREGION}
end;

end.
