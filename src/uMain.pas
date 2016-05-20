unit uMain;

interface

uses
  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants, System.Classes, Vcl.Graphics,
  Vcl.Controls, Vcl.Forms, Vcl.Dialogs, Vcl.StdCtrls, Vcl.Samples.Spin, Vcl.ExtCtrls,
  uNeuralNetworkBase;

type
  TForm1 = class(TForm)
    btnLoad: TButton;
    mmoLog: TMemo;
    dlgFiles: TOpenDialog;
    btnLearn: TButton;
    seEpochs: TSpinEdit;
    lblEpochs: TLabel;
    edtEta: TEdit;
    grpTopology: TGroupBox;
    seInput: TSpinEdit;
    seHidden: TSpinEdit;
    seOutput: TSpinEdit;
    lblInput: TLabel;
    lblHidden: TLabel;
    lblOutput: TLabel;
    lbledtData: TLabeledEdit;
    lbledtWeights: TLabeledEdit;
    btnData: TButton;
    btnWeights: TButton;
    lblEta: TLabel;
    lblEpochsComputed: TLabel;
    btnTests: TButton;
    btnGPU: TButton;
    btnKernels: TButton;
    procedure btnLoadClick(Sender: TObject);
    procedure btnDataClick(Sender: TObject);
    procedure btnWeightsClick(Sender: TObject);
    procedure btnTestsClick(Sender: TObject);
    procedure btnKernelsClick(Sender: TObject);
    procedure btnLearnClick(Sender: TObject);
    procedure btnGPUClick(Sender: TObject);
  private
    { Private declarations }
    procedure Learn(ANeuralNetwork: TNeuralNetworkBase);
  public
    { Public declarations }
  end;

var
  Form1: TForm1;

implementation

uses
  uSamples, uNeuralNetwork, uNeuralNetworkOpenCL;

{$R *.dfm}

procedure TForm1.btnDataClick(Sender: TObject);
begin
  if dlgFiles.Execute then
    lbledtData.Text := dlgFiles.FileName;
end;

procedure TForm1.btnGPUClick(Sender: TObject);
var
  Topology: TTopology;
  ANeuralNetwork: TNeuralNetworkBase;
begin
  Topology.Input := seInput.Value;
  Topology.Hidden := seHidden.Value;
  Topology.Output := seOutput.Value;

  try
    ANeuralNetwork := TNeuralNetworkOpenCL.Create(Topology);
    Learn(ANeuralNetwork);
  finally
    FreeAndNil(ANeuralNetwork);
  end;
end;

procedure TForm1.btnKernelsClick(Sender: TObject);
var
  NNOpenCL: TNeuralNetworkOpenCL;
  Topology: TTopology;
begin
  Topology.Input := seInput.Value;
  Topology.Hidden := seHidden.Value;
  Topology.Output := seOutput.Value;
  try
    NNOpenCL := TNeuralNetworkOpenCL.Create(Topology);
    NNOpenCL.Log := mmoLog.Lines;
    NNOpenCL.BuildKernel;
    NNOpenCL.Multiply;
    NNOpenCL.DeltaOutput;
    NNOpenCL.DeltaHidden;
    NNOpenCL.UpdateWeights;
  finally
    FreeAndNil(NNOpenCL);
  end;
end;

procedure TForm1.btnLearnClick(Sender: TObject);
var
  Topology: TTopology;
  ANeuralNetwork: TNeuralNetworkBase;
begin
  Topology.Input := seInput.Value;
  Topology.Hidden := seHidden.Value;
  Topology.Output := seOutput.Value;

  try
    ANeuralNetwork := TNeuralNetwork.Create(Topology);
    Learn(ANeuralNetwork);
  finally
    FreeAndNil(ANeuralNetwork);
  end;
end;

procedure TForm1.btnLoadClick(Sender: TObject);
var
  Samples: TSamplesSet;
  Row, Col: Cardinal;
  Line: string;
begin
  if not dlgFiles.Execute then
    Exit;

  mmoLog.Lines.Clear;

  try
    Samples :=  TSamplesSet.Create;
    Samples.LoadCSVFile(dlgFiles.FileName, 2, 1, ',');
    for Row := 0 to Samples.SamplesCount - 1 do
    begin
      Line := '';
      for Col := 0 to Samples.SampleSize - 1 do
        Line := Line + FloatToStr(Samples.Samples[Row][Col]) + ' | ';

      mmoLog.Lines.Add(Line);
    end;
  finally
    FreeAndNil(Samples);
  end;
end;

procedure TForm1.btnTestsClick(Sender: TObject);
const
  DELIMITER = ';';
var
  Net: TNeuralNetwork;
  Topology: TTopology;
  Samples: TSamplesSet;
begin
  Topology.Input := seInput.Value;
  Topology.Hidden := seHidden.Value;
  Topology.Output := seOutput.Value;

  try
    Samples := TSamplesSet.Create;
    Samples.LoadCSVFile(lbledtData.Text, Topology.Input, Topology.Output, DELIMITER);

    Net := TNeuralNetwork.Create(Topology);
    Net.Log := mmoLog.Lines;
    Net.LoadWeights(lbledtWeights.Text);

    mmoLog.Lines.Clear;
    mmoLog.Lines.BeginUpdate;

    Net.Tests(Samples);

    mmoLog.Lines.EndUpdate;
    //mmoLog.Lines.SaveToFile('D:\Libraries\Documents\GitHub\ParallelNeuralNetwork\data\tests.csv');
  finally
    FreeAndNil(Net);
    FreeAndNil(Samples);
  end;
end;

procedure TForm1.btnWeightsClick(Sender: TObject);
begin
  if dlgFiles.Execute then
    lbledtWeights.Text := dlgFiles.FileName;
end;

procedure TForm1.Learn(ANeuralNetwork: TNeuralNetworkBase);
const
  DELIMITER = ';';
var
  Topology: TTopology;
  Samples: TSamplesSet;
  Epochs: Integer;
  TickCount: Cardinal;
begin
  Topology.Input := seInput.Value;
  Topology.Hidden := seHidden.Value;
  Topology.Output := seOutput.Value;

  //mmoLog.Lines.Clear;

  //Epochs := seEpochs.Value;
  Epochs := 1;
  try
    Samples := TSamplesSet.Create;
    Samples.LoadCSVFile(lbledtData.Text, Topology.Input, Topology.Output, DELIMITER);

    ANeuralNetwork.Log := mmoLog.Lines;
    ANeuralNetwork.Eta := StrToFloat(edtEta.Text);
    ANeuralNetwork.LoadWeights('C:\Temp\weights_20160520-092204.csv');
    //ANeuralNetwork.DefineRandomWeights;

    //mmoLog.Lines.BeginUpdate;

//    for i := 0 to 9 do
//      mmoLog.Lines.Add(IntToStr(i));

    TickCount := TThread.GetTickCount;

    lblEpochsComputed.Tag := 0;
    for Epochs := Epochs downto 0 do
    begin
      ANeuralNetwork.Learn(Samples);

      lblEpochsComputed.Tag := lblEpochsComputed.Tag + 1;
      if Epochs mod 100 = 0 then
      begin
        lblEpochsComputed.Caption := 'Epochs Computed: ' + IntToStr(lblEpochsComputed.Tag);
        Application.ProcessMessages;
      end;
    end;

    TickCount := TThread.GetTickCount - TickCount;
    ShowMessage('TickCount = ' + IntToStr(TickCount));
    //mmoLog.Lines.Add('TickCount = ' + IntToStr(TickCount));

    ANeuralNetwork.SaveWeights(lbledtWeights.Text);

    //mmoLog.Lines.EndUpdate;
    //mmoLog.Lines.SaveToFile('D:\Libraries\Documents\GitHub\ParallelNeuralNetwork\data\trained.csv');
  finally
    FreeAndNil(Samples);
  end;
end;

end.
