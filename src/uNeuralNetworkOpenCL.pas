unit uNeuralNetworkOpenCL;

interface

uses
  Mitov.OpenCL, System.Classes, uNeuralNetworkBase, uSamples, uTypes;

type
  TNeuralNetworkOpenCL = class(TNeuralNetworkBase)
  protected
    FContext: IOCLContext;
    FCommandQueue: IOCLCommandQueue;
    FKernelFeedForward: IOCLKernel;
    FKernelDeltaOutput: IOCLKernel;
    FKernelDeltaHidden: IOCLKernel;
    FKernelUpdateWeights: IOCLKernel;

    FBufferSamples: IOCLBuffer;
    // o FBufferNeuronsInput foi substitu�do pelo FBufferSamples para evitar c�pia desnecess�ria de dados
    //FBufferNeuronsInput: IOCLBuffer;
    FBufferNeuronsHidden: IOCLBuffer;
    FBufferNeuronsOutput: IOCLBuffer;

    FBufferWInputHidden: IOCLBuffer;
    FBufferWHiddenOutput: IOCLBuffer;

    FBufferDeltaOutput: IOCLBuffer;
    FBufferDeltaHidden: IOCLBuffer;

    /// <summary>
    ///  Obt�m o c�digo fonte do Kernel OpenCL vinculado como recurso ao execut�vel.
    /// </summary>
    /// <param name="AResourceName">
    ///  Nome do recurso vinculado ao execut�vel.
    /// </param>
    /// <returns>
    ///  Texto do c�digo fonte.
    /// </returns>
    /// <remarks>
    ///  M�todo utilizado para compilar o kernel OpenCL.
    /// </remarks>
    function GetOpenCLSource(const AResourceName: string): string;
    /// <summary>
    ///  Selecione o device (GPU), carrega o c�digo fonte e compila os kernels OpenCL. Este m�todo prepara para a
    ///  execu��o dos kernels na GPU.
    /// </summary>
    procedure BuildKernel;
    /// <summary>
    ///  Cria os objetos que representam os dados que ser�o transferidos para a GPU.
    /// </summary>
    procedure CreateBuffers;
    /// <summary>
    ///  Envia os dados do HOST para a GPU.
    /// </summary>
    procedure WriteBufferToGPU;
    /// <summary>
    ///  Calcula a etapa de FeedForward do algoritmo de aprendizagem da rede neural. O c�lculo � realizado para todas
    ///  as camadas da rede neural, ou seja, Input -> Hidden e Hidden -> Output.
    /// </summary>
    /// <param name="iSample">
    ///  �ndice da amostra que est� sendo computada.
    /// </param>
    procedure FeedForward(iSample: Cardinal); override;
    /// <summary>
    ///  Calcula a etapa de backpropagation do algoritmo de aprendizagem da rede neural. Nesta etapa, � calculado o
    ///  Delta que representa o quanto a resposta est� diferente do esperado e depois utiliza este valor para atualizar
    ///  os pesos entre os neur�nios, iniciando na camada de sa�da at� a camada de entrada.
    /// </summary>
    /// <param name="iSample">
    ///  �ndice da amostra que est� sendo computada.
    /// </param>
    procedure BackPropagation(iSample: Cardinal); override;
  public
    /// <summary>
    ///  Salva os pesos das conex�o entre os neur�nios em arquivo, no formato CSV. Os pesos correspondem aos valores
    ///  utilizados no vetor 1D.
    /// </summary>
    /// <param name="AFileName">
    ///  Caminho completo do arquivo que ser� criado para salvar o valor dos pesos.
    /// </param>
    procedure SaveWeights(const AFileName: string); override;
    /// <summary>
    ///  Executa o m�todo BuildKernel.
    /// </summary>
    procedure Prepare; override;
    /// <summary>
    ///  Executa a etapa de aprendizagem, ou seja, computa o FeedForward e Backpropagation para todas as entradas do
    ///  conjunto de amostras, repetindo AEpochs vezes.
    /// </summary>
    /// <remarks>
    ///  A condi��o de parada do m�todo � computar todas as entradas do conjunto de amostras. N�o h� um controle de
    ///  parada com base na margem de erro do previsto e computado.
    /// </remarks>
    procedure Learn(AEpochs: Cardinal); override;
  end;

implementation

uses
  System.Types, System.SysUtils, uHelpers;

{ TNeuralNetworkOpenCL }

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

  // retira da lista todos os devices que n�o s�o GPU
  i := APlatform.Devices.Count - 1;
  while i >= 0 do
  begin
    if APlatform.Devices.Items[i].DeviceTypes <> [TOCLDeviceType.GPU] then
      APlatform.Devices.Delete(i);
    Dec(i);
  end;
  // cria o contexto apenas para os devices GPUs
  FContext := TOCLContext.Create(APlatform.Devices);

  // carrega o c�digo fonte do kernel OpenCL e compila para os DEVICES
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

  // obt�m a refer�ncia para os kernels
  FKernelFeedForward := AProgram.Kernel['feedForward'];
  FKernelDeltaOutput := AProgram.Kernel['calculateDeltaOutput'];
  FKernelDeltaHidden := AProgram.Kernel['calculateDeltaHidden'];
  FKernelUpdateWeights := AProgram.Kernel['updateWeights'];

  // cria a fila de execu��o dos kernels e demais comandos
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
  FCommandQueue.EnqueueReadBuffer(FBufferWInputHidden, True, @FWeights1DInputHidden[0]);
  FCommandQueue.EnqueueReadBuffer(FBufferWHiddenOutput, True, @FWeights1DHiddenOutput[0]);
  FCommandQueue.Finish;
end;

procedure TNeuralNetworkOpenCL.Prepare;
begin
  inherited;
  BuildKernel;
end;

procedure TNeuralNetworkOpenCL.SaveWeights(const AFileName: string);
var
  Weights: TStringList;
  Line: string;
  i, h, o, k: Integer;
begin
  try
    Weights := TStringList.Create;

    for i := 0 to FTopology.Input {+ BIAS} do
    begin
      k := i * FTopology.Hidden;
      Line := FloatToStr(FWeights1DInputHidden[k]);
      for h := 1 to FTopology.Hidden - 1 do
      begin
        k := i * FTopology.Hidden + h;
        Line := Line + ';' + FloatToStr(FWeights1DInputHidden[k]);
      end;

      Weights.Add(Line);
    end;

    for h := 0 to FTopology.Hidden {+ BIAS} do
    begin
      k := h * FTopology.Output;
      Line := FloatToStr(FWeights1DHiddenOutput[k]);
      for o := 1 to FTopology.Output - 1 do
      begin
        k := h * FTopology.Output + o;
        Line := Line + ';' + FloatToStr(FWeights1DHiddenOutput[k]);
      end;

      Weights.Add(Line);
    end;

    Weights.SaveToFile(AFileName);
  finally
    FreeAndNil(Weights);
  end;
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
  // o FBufferNeuronsInput foi substitu�do pelo FBufferSamples para evitar c�pia desnecess�ria de dados
  //FBufferNeuronsInput := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (FTopology.Input + 1) * SizeOf(Single), @FNeuronsInput[0]); // +1 for BIAS
  FBufferNeuronsHidden := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (FTopology.Hidden + 1) * SizeOf(Single), @FNeuronsHidden[0]); // +1 for BIAS
  FBufferNeuronsOutput := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], FTopology.Output * SizeOf(Single), @FNeuronsOutput[0]);

  FBufferWInputHidden := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (FTopology.Input + 1) * FTopology.Hidden * SizeOf(Single), @FWeights1DInputHidden[0]);
  FBufferWHiddenOutput := FContext.CreateBuffer([TOCLMemoryFlag.ReadWrite, TOCLMemoryFlag.UseHostPtr], (FTopology.Hidden + 1) * FTopology.Output * SizeOf(Single), @FWeights1DHiddenOutput[0]);

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
  // o FBufferNeuronsInput foi substitu�do pelo FBufferSamples para evitar c�pia desnecess�ria de dados
  //FCommandQueue.EnqueueWriteBuffer(FBufferNeuronsInput, True, @FNeuronsInput[0]);
  FCommandQueue.EnqueueWriteBuffer(FBufferNeuronsHidden, True, @FNeuronsHidden[0]);
  FCommandQueue.EnqueueWriteBuffer(FBufferNeuronsOutput, True, @FNeuronsOutput[0]);

  FCommandQueue.EnqueueWriteBuffer(FBufferWInputHidden, True, @FWeights1DInputHidden[0]);
  FCommandQueue.EnqueueWriteBuffer(FBufferWHiddenOutput, True, @FWeights1DHiddenOutput[0]);

  FCommandQueue.EnqueueWriteBuffer(FBufferDeltaOutput, True, @FDeltaOutput[0]);
  FCommandQueue.EnqueueWriteBuffer(FBufferDeltaHidden, True, @FDeltaHidden[0]);

  FCommandQueue.Flush;
  FCommandQueue.Finish;

  TickCount := TThread.GetTickCount - TickCount;
  FLog.AddFmt('Elapsed time: %d ms', [TickCount]);
end;

procedure TNeuralNetworkOpenCL.FeedForward(iSample: Cardinal);
var
  M, N, K: Integer;
  InputOffSet: Integer;
begin
  {$REGION 'Calcular ativa��o INPUT --> HIDDEN'}
  M := 1; // 1 amostra
  N := FTopology.Hidden;
  K := FTopology.Input + 1; // +1 for BIAS
  InputOffSet := iSample * (FTopology.Input + 1 + FTopology.Output); // +1 for BIAS

  FKernelFeedForward.Arguments[0].Access.SetValue<Integer>(M);
  FKernelFeedForward.Arguments[1].Access.SetValue<Integer>(N);
  FKernelFeedForward.Arguments[2].Access.SetValue<Integer>(K);
  FKernelFeedForward.Arguments[3].Access.SetValue<Integer>(InputOffSet);
  FKernelFeedForward.Arguments[4].Access.SetBuffer(FBufferSamples);
  FKernelFeedForward.Arguments[5].Access.SetBuffer(FBufferWInputHidden);
  FKernelFeedForward.Arguments[6].Access.SetBuffer(FBufferNeuronsHidden);

  FCommandQueue.EnqueueNDRangeKernel(FKernelFeedForward, TOCLGlobalDimensions.Create([M, N]));
  {$ENDREGION}

  {$REGION 'Calcular ativa��o HIDDEN --> OUTPUT'}
  M := 1; // 1 amostra
  N := FTopology.Output;
  K := FTopology.Hidden + 1; // +1 for BIAS

  FKernelFeedForward.Arguments[0].Access.SetValue<Integer>(M);
  FKernelFeedForward.Arguments[1].Access.SetValue<Integer>(N);
  FKernelFeedForward.Arguments[2].Access.SetValue<Integer>(K);
  FKernelFeedForward.Arguments[3].Access.SetValue<Integer>(0);
  FKernelFeedForward.Arguments[4].Access.SetBuffer(FBufferNeuronsHidden);
  FKernelFeedForward.Arguments[5].Access.SetBuffer(FBufferWHiddenOutput);
  FKernelFeedForward.Arguments[6].Access.SetBuffer(FBufferNeuronsOutput);

  FCommandQueue.EnqueueNDRangeKernel(FKernelFeedForward, TOCLGlobalDimensions.Create([M, N]));
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
