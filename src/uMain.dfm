object Form1: TForm1
  Left = 0
  Top = 0
  Caption = 'Parallel Neural Network'
  ClientHeight = 777
  ClientWidth = 615
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
  object mmoLog: TMemo
    Left = 8
    Top = 200
    Width = 596
    Height = 569
    ScrollBars = ssBoth
    TabOrder = 0
  end
  object btnSequential: TButton
    Left = 530
    Top = 102
    Width = 75
    Height = 25
    Caption = 'Sequential'
    TabOrder = 1
    OnClick = btnSequentialClick
  end
  object seEpochs: TSpinEdit
    Left = 344
    Top = 27
    Width = 98
    Height = 22
    MaxValue = 0
    MinValue = 0
    TabOrder = 2
    Value = 1000
  end
  object edtEta: TEdit
    Left = 208
    Top = 27
    Width = 121
    Height = 21
    TabOrder = 3
    Text = '0,15'
  end
  object grpTopology: TGroupBox
    Left = 8
    Top = 8
    Width = 185
    Height = 73
    Caption = ' Topology '
    TabOrder = 4
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
      Value = 11
    end
    object seHidden: TSpinEdit
      Left = 67
      Top = 36
      Width = 50
      Height = 22
      MaxValue = 10000
      MinValue = 1
      TabOrder = 1
      Value = 20
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
    Width = 473
    Height = 21
    EditLabel.Width = 40
    EditLabel.Height = 13
    EditLabel.Caption = 'Data file'
    TabOrder = 5
    Text = 
      'D:\Libraries\Documents\GitHub\ParallelNeuralNetwork\data\Wine\wi' +
      'ne-red-normal.csv'
  end
  object lbledtWeights: TLabeledEdit
    Left = 8
    Top = 145
    Width = 473
    Height = 21
    EditLabel.Width = 56
    EditLabel.Height = 13
    EditLabel.Caption = 'Weights file'
    TabOrder = 6
    Text = 
      'D:\Libraries\Documents\GitHub\ParallelNeuralNetwork\data\Wine\wi' +
      'ne-red-weights.csv'
  end
  object btnData: TButton
    Left = 480
    Top = 103
    Width = 23
    Height = 23
    Caption = '...'
    TabOrder = 7
    OnClick = btnDataClick
  end
  object btnWeights: TButton
    Left = 480
    Top = 143
    Width = 23
    Height = 23
    Caption = '...'
    TabOrder = 8
    OnClick = btnWeightsClick
  end
  object btnTests: TButton
    Left = 529
    Top = 144
    Width = 75
    Height = 25
    Caption = 'Tests'
    TabOrder = 9
    OnClick = btnTestsClick
  end
  object btnParallel: TButton
    Left = 532
    Top = 42
    Width = 75
    Height = 25
    Caption = 'Parallel'
    TabOrder = 10
    OnClick = btnParallelClick
  end
  object dlgFiles: TOpenDialog
    Filter = 'CSV File (*.csv)|*.csv'
    Left = 472
    Top = 24
  end
end
