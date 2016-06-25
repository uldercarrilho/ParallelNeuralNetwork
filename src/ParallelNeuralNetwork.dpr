program ParallelNeuralNetwork;

{$R 'Kernels.res' 'Kernels.rc'}

uses
  Vcl.Forms,
  uMain in 'uMain.pas' {Form1},
  uSamples in 'uSamples.pas',
  uNeuralNetworkSequential in 'uNeuralNetworkSequential.pas',
  uNeuralNetworkOpenCL in 'uNeuralNetworkOpenCL.pas',
  uNeuralNetworkBase in 'uNeuralNetworkBase.pas',
  uHelpers in 'uHelpers.pas',
  uTypes in 'uTypes.pas',
  uNeuralNetworkOpenCLTests in 'uNeuralNetworkOpenCLTests.pas';

{$R *.res}

begin
  Application.Initialize;
  Application.MainFormOnTaskbar := True;
  Application.CreateForm(TForm1, Form1);
  Application.Run;
end.
