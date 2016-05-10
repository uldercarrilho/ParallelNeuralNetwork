unit uMain;

interface

uses
  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants, System.Classes, Vcl.Graphics,
  Vcl.Controls, Vcl.Forms, Vcl.Dialogs, Vcl.StdCtrls, Vcl.Samples.Spin, Vcl.ExtCtrls;

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
    procedure btnLoadClick(Sender: TObject);
    procedure btnDataClick(Sender: TObject);
    procedure btnWeightsClick(Sender: TObject);
    procedure btnLearnClick(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  Form1: TForm1;

implementation

uses
  uSamples, uNeuralNetwork;

{$R *.dfm}

procedure TForm1.btnDataClick(Sender: TObject);
begin
  if dlgFiles.Execute then
    lbledtData.Text := dlgFiles.FileName;
end;

procedure TForm1.btnLearnClick(Sender: TObject);
const
  DELIMITER = ';';
var
  Net: TNeuralNetwork;
  Topology: TTopology;
  Samples: TSamplesSet;
  Epochs: Integer;
  TickCount: Cardinal;
  i: Integer;
begin
  Topology.Input := seInput.Value;
  Topology.Hidden := seHidden.Value;
  Topology.Output := seOutput.Value;

  Epochs := seEpochs.Value;

  try
    Samples := TSamplesSet.Create;
    Samples.LoadCSVFile(lbledtData.Text, Topology.Input, Topology.Output, DELIMITER);

    Net := TNeuralNetwork.Create(Topology);
    Net.Log := mmoLog.Lines;
    Net.Eta := StrToFloat(edtEta.Text);
    Net.DefineRandomWeights;

    mmoLog.Lines.Clear;
    mmoLog.Lines.BeginUpdate;

    for i := 0 to Samples.SamplesCount - 1 do
      mmoLog.Lines.Add(IntToStr(i));

    TickCount := TThread.GetTickCount;

    lblEpochsComputed.Tag := 0;
    for Epochs := Epochs downto 0 do
    begin
      Net.Learn(Samples);

      lblEpochsComputed.Tag := lblEpochsComputed.Tag + 1;
      lblEpochsComputed.Caption := 'Epochs Computed: ' + IntToStr(lblEpochsComputed.Tag);
      Application.ProcessMessages;
    end;

    TickCount := TThread.GetTickCount - TickCount;
    ShowMessage('TickCount = ' + IntToStr(TickCount));
    //mmoLog.Lines.Add('TickCount = ' + IntToStr(TickCount));

    mmoLog.Lines.EndUpdate;
    mmoLog.Lines.SaveToFile('D:\Libraries\Documents\GitHub\ParallelNeuralNetwork\data\trained.csv');
  finally
    FreeAndNil(Net);
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

procedure TForm1.btnWeightsClick(Sender: TObject);
begin
  if dlgFiles.Execute then
    lbledtWeights.Text := dlgFiles.FileName;
end;

end.