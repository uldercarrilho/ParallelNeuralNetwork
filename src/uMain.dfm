object Form1: TForm1
  Left = 0
  Top = 0
  BorderIcons = [biSystemMenu, biMinimize]
  BorderStyle = bsSingle
  Caption = 'Parallel Neural Network'
  ClientHeight = 471
  ClientWidth = 965
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
  object lblEpochsComputed: TLabel
    Left = 495
    Top = 8
    Width = 35
    Height = 13
    Caption = 'Results'
  end
  object mmoLog: TMemo
    Left = 495
    Top = 27
    Width = 458
    Height = 432
    ScrollBars = ssBoth
    TabOrder = 3
  end
  object grpTopology: TGroupBox
    Left = 8
    Top = 8
    Width = 481
    Height = 73
    Caption = ' Topology '
    TabOrder = 0
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
  object grpTests: TGroupBox
    Left = 8
    Top = 319
    Width = 481
    Height = 140
    Caption = ' Tests '
    TabOrder = 2
    object edtTestsData: TLabeledEdit
      Left = 17
      Top = 36
      Width = 424
      Height = 21
      EditLabel.Width = 40
      EditLabel.Height = 13
      EditLabel.Caption = 'Data file'
      TabOrder = 0
    end
    object btnTestsData: TButton
      Left = 441
      Top = 35
      Width = 23
      Height = 23
      Caption = '...'
      TabOrder = 1
      OnClick = btnTestsDataClick
    end
    object btnTests: TButton
      Left = 366
      Top = 104
      Width = 75
      Height = 25
      Caption = 'Tests'
      TabOrder = 4
      OnClick = btnTestsClick
    end
    object edtTestsWeights: TLabeledEdit
      Left = 17
      Top = 76
      Width = 424
      Height = 21
      EditLabel.Width = 51
      EditLabel.Height = 13
      EditLabel.Caption = 'Weight file'
      TabOrder = 2
    end
    object btnTestsWeights: TButton
      Left = 441
      Top = 75
      Width = 23
      Height = 23
      Caption = '...'
      TabOrder = 3
      OnClick = btnTestsWeightsClick
    end
  end
  object grpTraining: TGroupBox
    Left = 8
    Top = 87
    Width = 481
    Height = 226
    Caption = ' Training '
    TabOrder = 1
    object lblEta: TLabel
      Left = 17
      Top = 23
      Width = 94
      Height = 13
      Caption = 'Learning coefficient'
    end
    object lblEpochs: TLabel
      Left = 125
      Top = 22
      Width = 34
      Height = 13
      Caption = 'Epochs'
    end
    object edtEta: TEdit
      Left = 17
      Top = 42
      Width = 94
      Height = 21
      TabOrder = 0
      Text = '0,15'
    end
    object seEpochs: TSpinEdit
      Left = 125
      Top = 41
      Width = 60
      Height = 22
      MaxValue = 0
      MinValue = 0
      TabOrder = 1
      Value = 100
    end
    object btnTrainingData: TButton
      Left = 441
      Top = 90
      Width = 23
      Height = 23
      Caption = '...'
      TabOrder = 3
      OnClick = btnTrainingDataClick
    end
    object edtTrainingData: TLabeledEdit
      Left = 17
      Top = 91
      Width = 424
      Height = 21
      EditLabel.Width = 40
      EditLabel.Height = 13
      EditLabel.Caption = 'Data file'
      TabOrder = 2
    end
    object rbRandomWeights: TRadioButton
      Left = 17
      Top = 118
      Width = 113
      Height = 17
      Caption = 'Random weights'
      Checked = True
      TabOrder = 4
      TabStop = True
    end
    object rbWeightsFromFile: TRadioButton
      Left = 17
      Top = 141
      Width = 113
      Height = 17
      Caption = 'Weights from file'
      TabOrder = 5
    end
    object btnTrainingWeights: TButton
      Left = 441
      Top = 163
      Width = 23
      Height = 23
      Caption = '...'
      TabOrder = 7
      OnClick = btnTrainingWeightsClick
    end
    object edtTrainingWeights: TEdit
      Left = 32
      Top = 164
      Width = 409
      Height = 21
      TabOrder = 6
    end
    object btnParallel: TButton
      Left = 366
      Top = 192
      Width = 75
      Height = 25
      Caption = 'Parallel'
      TabOrder = 9
      OnClick = btnParallelClick
    end
    object btnSequential: TButton
      Left = 285
      Top = 192
      Width = 75
      Height = 25
      Caption = 'Sequential'
      TabOrder = 8
      OnClick = btnSequentialClick
    end
  end
  object dlgFiles: TOpenDialog
    Filter = 'CSV File (*.csv)|*.csv'
    Left = 296
    Top = 32
  end
end
