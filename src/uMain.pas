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
    edtTrainingData: TLabeledEdit;
    btnTrainingData: TButton;
    btnTrainingWeights: TButton;
    lblEta: TLabel;
    lblEpochsComputed: TLabel;
    btnTests: TButton;
    btnParallel: TButton;
    grpTraining: TGroupBox;
    rbRandomWeights: TRadioButton;
    rbWeightsFromFile: TRadioButton;
    edtTrainingWeights: TEdit;
    grpTests: TGroupBox;
    edtTestsData: TLabeledEdit;
    btnTestsData: TButton;
    edtTestsWeights: TLabeledEdit;
    btnTestsWeights: TButton;
    procedure btnTrainingDataClick(Sender: TObject);
    procedure btnTrainingWeightsClick(Sender: TObject);
    procedure btnTestsClick(Sender: TObject);
    procedure btnSequentialClick(Sender: TObject);
    procedure btnParallelClick(Sender: TObject);
    procedure btnTestsDataClick(Sender: TObject);
    procedure btnTestsWeightsClick(Sender: TObject);
  private
    { Private declarations }
    procedure Learn(ANeuralNetworkClass: TNeuralNetworkBaseClass);
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

procedure TForm1.btnTrainingDataClick(Sender: TObject);
begin
  if dlgFiles.Execute then
    edtTrainingData.Text := dlgFiles.FileName;
end;

procedure TForm1.btnTrainingWeightsClick(Sender: TObject);
begin
  if dlgFiles.Execute then
    edtTrainingWeights.Text := dlgFiles.FileName;
end;

procedure TForm1.btnTestsDataClick(Sender: TObject);
begin
  if dlgFiles.Execute then
    edtTestsData.Text := dlgFiles.FileName;
end;

procedure TForm1.btnTestsWeightsClick(Sender: TObject);
begin
  if dlgFiles.Execute then
    edtTestsWeights.Text := dlgFiles.FileName;
end;

procedure TForm1.btnParallelClick(Sender: TObject);
begin
  mmoLog.Lines.Clear;
  mmoLog.Lines.Add('Running parallel algorithm on GPU');
  Learn(TNeuralNetworkOpenCL);
end;

procedure TForm1.btnSequentialClick(Sender: TObject);
begin
  mmoLog.Lines.Clear;
  mmoLog.Lines.Add('Running sequential algorithm on CPU');
  Learn(TNeuralNetworkSequential);
end;

procedure TForm1.Learn(ANeuralNetworkClass: TNeuralNetworkBaseClass);
var
  Topology: TTopology;
  Samples: TSamplesSet;
  NeuralNetwork: TNeuralNetworkBase;
  TickCount: Cardinal;
  Filename: TFileName;
begin
  Topology.Input := seInput.Value;
  Topology.Hidden := seHidden.Value;
  Topology.Output := seOutput.Value;

  Samples := TSamplesSet.Create;
  try
    Samples.LoadCSVFile(edtTrainingData.Text, Topology.Input, Topology.Output, DELIMITER);

    // configura a rede neural
    NeuralNetwork := ANeuralNetworkClass.Create(Topology);
    NeuralNetwork.Log := mmoLog.Lines;
    NeuralNetwork.Eta := StrToFloat(edtEta.Text);
    NeuralNetwork.SamplesSet := Samples;

    if rbRandomWeights.Checked then
    begin
      NeuralNetwork.DefineRandomWeights;
      Filename := ExtractFilePath(edtTrainingData.Text) + 'RandomWeights_' + FormatDateTime('yyyymmdd-hhnnss', Now) + '.csv';
      NeuralNetwork.SaveWeights(Filename);
    end
    else
      NeuralNetwork.LoadWeights(edtTrainingWeights.Text);

    NeuralNetwork.Prepare;

    // treinamento da rede neural
    TickCount := TThread.GetTickCount;
    NeuralNetwork.Learn(seEpochs.Value);
    TickCount := TThread.GetTickCount - TickCount;
    mmoLog.Lines.AddFmt('Total training time: %d ms ', [TickCount]);

    // armazena os pesos atualizados
    Filename := ExtractFilePath(edtTrainingData.Text) + 'weights_' + FormatDateTime('yyyymmdd-hhnnss', Now) + '.csv';
    NeuralNetwork.SaveWeights(Filename);
    mmoLog.Lines.Add('Weights saved in ' + Filename);
  finally
    FreeAndNil(Samples);
  end;
end;

procedure TForm1.btnTestsClick(Sender: TObject);
var
  NeuralNetwork: TNeuralNetworkSequential;
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
    Samples.LoadCSVFile(edtTestsData.Text, Topology.Input, Topology.Output, DELIMITER);

    NeuralNetwork := TNeuralNetworkSequential.Create(Topology);
    NeuralNetwork.Log := mmoLog.Lines;
    NeuralNetwork.LoadWeights(edtTestsWeights.Text);
    NeuralNetwork.SamplesSet := Samples;

    NeuralNetwork.Tests;

    Filename := ExtractFilePath(edtTestsData.Text) + 'tests_' + FormatDateTime('yyyymmdd-hhnnss', Now) + '.csv';
    mmoLog.Lines.SaveToFile(Filename);
  finally
    mmoLog.Lines.EndUpdate;
    FreeAndNil(NeuralNetwork);
    FreeAndNil(Samples);
  end;
end;

end.
