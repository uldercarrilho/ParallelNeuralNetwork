unit uSamples;

interface

type
  PSample = ^TSample;
  TSample = array of Single;
  TSamples = array of TSample;

  TSamplesSet = class
  private
    FInputSize: Word;
    FOutputSize: Word;
    FSampleSize: Cardinal;
    FSamples: TSamples;
    FSamplesCount: Cardinal;
  public
    FRaw: array of Single;

    constructor Create;
    destructor Destroy; override;

    procedure LoadCSVFile(const AFileName: string; const AInputSize, AOutputSize: Word; const ADelimiter: Char);

    property InputSize: Word read FInputSize;
    property OutputSize: Word read FOutputSize;
    property SampleSize: Cardinal read FSampleSize;
    property Samples: TSamples read FSamples;
    property SamplesCount: Cardinal read FSamplesCount;

  end;

implementation

uses
  System.SysUtils, System.Classes;

{ TSamplesSet }

constructor TSamplesSet.Create;
begin
  FInputSize := 0;
  FOutputSize := 0;
  FSampleSize := 0;
  FSamplesCount := 0;
end;

destructor TSamplesSet.Destroy;
begin
  SetLength(FSamples, 0);
  inherited;
end;

procedure TSamplesSet.LoadCSVFile(const AFileName: string; const AInputSize, AOutputSize: Word; const ADelimiter: Char);
var
  Row, Col: Cardinal;
  CSVFile: TStringList;
  Line: TStringList;
  i: Integer;
begin
  // TODO : lançar exceção
  if not FileExists(AFileName) then
    Exit;

  // TODO : lançar exceção
  if ADelimiter = '' then
    Exit;

  FInputSize := AInputSize;
  FOutputSize := AOutputSize;
  FSampleSize := AInputSize + AOutputSize;

  CSVFile := TStringList.Create;
  Line := TStringList.Create;
  try
    Line.Delimiter := ADelimiter;

    CSVFile.LoadFromFile(AFileName);

    FSamplesCount := CSVFile.Count;

    SetLength(FSamples, FSamplesCount, FSampleSize);
    SetLength(FRaw, FSamplesCount * FSampleSize);

    i := 0;
    for Row := 0 to FSamplesCount - 1 do
    begin
      Line.DelimitedText := CSVFile.Strings[Row];

      for Col := 0 to FSampleSize - 1 do
      begin
        FSamples[Row][Col] := StrToFloat(Line.Strings[Col]);
        FRaw[i] := FSamples[Row][Col];
        Inc(i);
      end;
    end;
  finally
    FreeAndNil(Line);
    FreeAndNil(CSVFile);
  end;
end;

end.
