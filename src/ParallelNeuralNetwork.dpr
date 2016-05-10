program ParallelNeuralNetwork;

uses
  Vcl.Forms,
  uMain in 'uMain.pas' {Form1},
  uSamples in 'uSamples.pas',
  uNeuralNetwork in 'uNeuralNetwork.pas';

{$R *.res}

begin
  Application.Initialize;
  Application.MainFormOnTaskbar := True;
  Application.CreateForm(TForm1, Form1);
  Application.Run;
end.
