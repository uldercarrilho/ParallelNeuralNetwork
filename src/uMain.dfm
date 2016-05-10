object Form1: TForm1
  Left = 0
  Top = 0
  Caption = 'Parallel Neural Network'
  ClientHeight = 777
  ClientWidth = 623
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -11
  Font.Name = 'Tahoma'
  Font.Style = []
  OldCreateOrder = False
  Position = poScreenCenter
  PixelsPerInch = 96
  TextHeight = 13
  object lblEpochs: TLabel
    Left = 344
    Top = 8
    Width = 34
    Height = 13
    Caption = 'Epochs'
  end
  object lblEta: TLabel
    Left = 208
    Top = 8
    Width = 94
    Height = 13
    Caption = 'Learning coefficient'
  end
  object lblEpochsComputed: TLabel
    Left = 8
    Top = 176
    Width = 99
    Height = 13
    Caption = 'Epochs Computed: 0'
  end
  object btnLoad: TButton
    Left = 448
    Top = 102
    Width = 75
    Height = 25
    Caption = 'Load'
    TabOrder = 0
    OnClick = btnLoadClick
  end
  object mmoLog: TMemo
    Left = 8
    Top = 200
    Width = 596
    Height = 569
    ScrollBars = ssBoth
    TabOrder = 1
  end
  object btnLearn: TButton
    Left = 529
    Top = 102
    Width = 75
    Height = 25
    Caption = 'Learn'
    TabOrder = 2
    OnClick = btnLearnClick
  end
  object seEpochs: TSpinEdit
    Left = 344
    Top = 27
    Width = 98
    Height = 22
    MaxValue = 0
    MinValue = 0
    TabOrder = 3
    Value = 1000
  end
  object edtEta: TEdit
    Left = 208
    Top = 27
    Width = 121
    Height = 21
    TabOrder = 4
    Text = '0,35'
  end
  object grpTopology: TGroupBox
    Left = 8
    Top = 8
    Width = 185
    Height = 73
    Caption = ' Topology '
    TabOrder = 5
    object lblInput: TLabel
      Left = 11
      Top = 17
      Width = 26
      Height = 13
      Caption = 'Input'
    end
    object lblHidden: TLabel
      Left = 67
      Top = 17
      Width = 33
      Height = 13
      Caption = 'Hidden'
    end
    object lblOutput: TLabel
      Left = 123
      Top = 17
      Width = 34
      Height = 13
      Caption = 'Output'
    end
    object seInput: TSpinEdit
      Left = 11
      Top = 36
      Width = 50
      Height = 22
      MaxValue = 10000
      MinValue = 1
      TabOrder = 0
      Value = 2
    end
    object seHidden: TSpinEdit
      Left = 67
      Top = 36
      Width = 50
      Height = 22
      MaxValue = 10000
      MinValue = 1
      TabOrder = 1
      Value = 3
    end
    object seOutput: TSpinEdit
      Left = 123
      Top = 36
      Width = 50
      Height = 22
      MaxValue = 10000
      MinValue = 1
      TabOrder = 2
      Value = 1
    end
  end
  object lbledtData: TLabeledEdit
    Left = 8
    Top = 104
    Width = 400
    Height = 21
    EditLabel.Width = 40
    EditLabel.Height = 13
    EditLabel.Caption = 'Data file'
    TabOrder = 6
    Text = 
      'D:\Libraries\Documents\GitHub\ParallelNeuralNetwork\data\samples' +
      '-XOR-2x1.csv'
  end
  object lbledtWeights: TLabeledEdit
    Left = 8
    Top = 144
    Width = 400
    Height = 21
    EditLabel.Width = 56
    EditLabel.Height = 13
    EditLabel.Caption = 'Weights file'
    TabOrder = 7
  end
  object btnData: TButton
    Left = 408
    Top = 103
    Width = 23
    Height = 23
    Caption = '...'
    TabOrder = 8
    OnClick = btnDataClick
  end
  object btnWeights: TButton
    Left = 408
    Top = 143
    Width = 23
    Height = 23
    Caption = '...'
    TabOrder = 9
    OnClick = btnWeightsClick
  end
  object dlgFiles: TOpenDialog
    Filter = 'CSV File (*.csv)|*.csv'
    Left = 472
    Top = 24
  end
end
