unit uMain;

interface

uses
  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants, System.Classes, Vcl.Graphics,
  Vcl.Controls, Vcl.Forms, Vcl.Dialogs, Vcl.StdCtrls, Vcl.Samples.Spin, Vcl.ExtCtrls,
  uNeuralNetworkBase;

type
  TForm1 = class(TForm)
    mmoLog: TMemo;
    dlgFiles: TOpenDialog;
    btnSequential: TButton;
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
    btnParallel: TButton;
    procedure btnDataClick(Sender: TObject);
    procedure btnWeightsClick(Sender: TObject);
    procedure btnTestsClick(Sender: TObject);
    procedure btnSequentialClick(Sender: TObject);
    procedure btnParallelClick(Sender: TObject);
  private
    { Private declarations }
    procedure Learn(ANeuralNetwork: TNeuralNetworkBase);
    procedure TestKernel;
  public
    { Public declarations }
  end;

var
  Form1: TForm1;

implementation

uses
  uSamples, uHelpers, uNeuralNetworkSequential, uNeuralNetworkOpenCL, uNeuralNetworkOpenCLTests;

const
  DELIMITER = ';';

{$R *.dfm}

procedure TForm1.btnDataClick(Sender: TObject);
begin
  if dlgFiles.Execute then
    lbledtData.Text := dlgFiles.FileName;
end;

procedure TForm1.btnWeightsClick(Sender: TObject);
begin
  if dlgFiles.Execute then
    lbledtWeights.Text := dlgFiles.FileName;
end;

procedure TForm1.btnParallelClick(Sender: TObject);
var
  Topology: TTopology;
  Samples: TSamplesSet;
  TickCount: Cardinal;
  NeuralNetwork: TNeuralNetworkOpenCL;
begin
  mmoLog.Lines.Add('Running parallel algorithm on GPU');

  Topology.Input := seInput.Value;
  Topology.Hidden := seHidden.Value;
  Topology.Output := seOutput.Value;

  Samples := TSamplesSet.Create;
  try
    Samples.LoadCSVFile(lbledtData.Text, Topology.Input, Topology.Output, DELIMITER);

    mmoLog.Lines.BeginUpdate;

    NeuralNetwork := TNeuralNetworkOpenCL.Create(Topology);
    NeuralNetwork.Log := mmoLog.Lines;
    NeuralNetwork.Eta := StrToFloat(edtEta.Text);
    NeuralNetwork.SetSamples(Samples.Samples1D, Samples.SamplesCount);
    NeuralNetwork.LoadWeights('C:\Temp\weights_20160524-204255.csv'); // wine-red 11x20x1
    //NeuralNetwork.DefineRandomWeights;



    TickCount := TThread.GetTickCount;

    NeuralNetwork.Learn(seEpochs.Value);

    TickCount := TThread.GetTickCount - TickCount;

    mmoLog.Lines.AddFmt('Learn time: %d ms ', [TickCount]);
    mmoLog.Lines.Add('');

    //NeuralNetwork.SaveWeights(lbledtWeights.Text);

    //mmoLog.Lines.SaveToFile('D:\Libraries\Documents\GitHub\ParallelNeuralNetwork\data\trained.csv');
  finally
    mmoLog.Lines.EndUpdate;
    FreeAndNil(Samples);
  end;
end;

procedure TForm1.btnSequentialClick(Sender: TObject);
var
  Topology: TTopology;
  ANeuralNetwork: TNeuralNetworkBase;
begin
  mmoLog.Lines.Clear;
  mmoLog.Lines.Add('Running sequential algorithm on CPU');
  Topology.Input := seInput.Value;
  Topology.Hidden := seHidden.Value;
  Topology.Output := seOutput.Value;

  try
    ANeuralNetwork := TNeuralNetworkSequential.Create(Topology);
    Learn(ANeuralNetwork);
  finally
    FreeAndNil(ANeuralNetwork);
  end;
end;

procedure TForm1.btnTestsClick(Sender: TObject);
var
  Net: TNeuralNetworkSequential;
  Topology: TTopology;
  Samples: TSamplesSet;
  Filename: TFileName;
begin
  mmoLog.Lines.Clear;

  Topology.Input := seInput.Value;
  Topology.Hidden := seHidden.Value;
  Topology.Output := seOutput.Value;

  Samples := TSamplesSet.Create;
  try
    mmoLog.Lines.BeginUpdate;
    Samples.LoadCSVFile(lbledtData.Text, Topology.Input, Topology.Output, DELIMITER);

    Net := TNeuralNetworkSequential.Create(Topology);
    Net.Log := mmoLog.Lines;
    Net.LoadWeights(lbledtWeights.Text);
    Net.SamplesSet := Samples;

    Net.Tests;

    Filename := ExtractFilePath(lbledtData.Text) + 'tests_' + FormatDateTime('yyyymmdd-hhnnss', Now) + '.csv';
    mmoLog.Lines.SaveToFile(Filename);
  finally
    mmoLog.Lines.EndUpdate;
    FreeAndNil(Net);
    FreeAndNil(Samples);
  end;
end;

procedure TForm1.Learn(ANeuralNetwork: TNeuralNetworkBase);
var
  Topology: TTopology;
  Samples: TSamplesSet;
  Epochs: Integer;
  TickCount: Cardinal;
  Filename: TFileName;
begin
  Topology.Input := seInput.Value;
  Topology.Hidden := seHidden.Value;
  Topology.Output := seOutput.Value;

  //mmoLog.Lines.Clear;

  Epochs := seEpochs.Value;
  //Epochs := 10;
  try
    Samples := TSamplesSet.Create;
    Samples.LoadCSVFile(lbledtData.Text, Topology.Input, Topology.Output, DELIMITER);

    ANeuralNetwork.Log := mmoLog.Lines;
    ANeuralNetwork.Eta := StrToFloat(edtEta.Text);
    ANeuralNetwork.SamplesSet := Samples;
    //ANeuralNetwork.LoadWeights('C:\Temp\weights_20160524-204255.csv'); // wine-red 11x20x1
    ANeuralNetwork.DefineRandomWeights;
    Filename := ExtractFilePath(ParamStr(0)) + 'weights_' + FormatDateTime('yyyymmdd-hhnnss', Now) + '.csv';
    ANeuralNetwork.SaveWeights(Filename);


    mmoLog.Lines.BeginUpdate;

//    for i := 0 to 9 do
//      mmoLog.Lines.Add(IntToStr(i));

    TickCount := TThread.GetTickCount;

    lblEpochsComputed.Tag := 0;
    for Epochs := Epochs downto 0 do
    begin
      ANeuralNetwork.Learn;

      lblEpochsComputed.Tag := lblEpochsComputed.Tag + 1;
      if Epochs mod 10 = 0 then
      begin
        lblEpochsComputed.Caption := 'Epochs Computed: ' + IntToStr(lblEpochsComputed.Tag);
        Application.ProcessMessages;
      end;
    end;

    TickCount := TThread.GetTickCount - TickCount;
    //ShowMessage('TickCount = ' + IntToStr(TickCount));
    mmoLog.Lines.Add('TickCount = ' + IntToStr(TickCount));
    mmoLog.Lines.Add('');

    ANeuralNetwork.SaveWeights(lbledtWeights.Text);

    mmoLog.Lines.EndUpdate;
    //mmoLog.Lines.SaveToFile('D:\Libraries\Documents\GitHub\ParallelNeuralNetwork\data\trained.csv');
  finally
    FreeAndNil(Samples);
  end;
end;

procedure TForm1.TestKernel;
var
  NNOpenCL: TNeuralNetworkOpenCLTests;
  Topology: TTopology;
begin
  // TODO : refatorar método para testar individualmente cada funcionalidade do Kernel

  Topology.Input := seInput.Value;
  Topology.Hidden := seHidden.Value;
  Topology.Output := seOutput.Value;
  try
    NNOpenCL := TNeuralNetworkOpenCLTests.Create(Topology);
    NNOpenCL.Log := mmoLog.Lines;
//    NNOpenCL.BuildKernel;
//    NNOpenCL.Multiply;
//    NNOpenCL.DeltaOutput;
//    NNOpenCL.DeltaHidden;
//    NNOpenCL.UpdateWeights;
//    NNOpenCL.Params;
  finally
    FreeAndNil(NNOpenCL);
  end;
end;

end.
