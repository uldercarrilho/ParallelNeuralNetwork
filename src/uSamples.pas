unit uSamples;

interface

uses
  uTypes;

type
  TSamplesSet = class
  private
    FInputSize: Word;
    FOutputSize: Word;
    FSampleSize: Cardinal;  // FInputSize + FOutputSize
    FSamples1D: TVector1D;  // usado para "vetorizar" os dados de FSamples2D e enviá-los à GPU
    FSamples2D: TVector2D;
    FSamplesCount: Cardinal;
  public
    constructor Create;
    destructor Destroy; override;
    /// <summary>
    ///  Carrega os dados em memória do arquivo informado no parâmetro AFileName.
    /// </summary>
    /// <param name="AFileName">
    ///  Nome do arquivo onde estão os dados estão armazenados no formato CSV.
    /// </param>
    /// <param name="AInputSize">
    ///  Tamanho (quantidade de colunas) da camada de entrada.
    /// </param>
    /// <param name="AOutputSize">
    ///  Tamanho (quantidade de colunas) da camada de saída.
    /// </param>
    /// <param name="ADelimiter">
    ///  Delimitador dos valores.
    /// </param>
    procedure LoadCSVFile(const AFileName: string; const AInputSize, AOutputSize: Word; const ADelimiter: Char);

    property InputSize: Word read FInputSize;
    property OutputSize: Word read FOutputSize;
    property SampleSize: Cardinal read FSampleSize;
    property Samples1D: TVector1D read FSamples1D;
    property Samples2D: TVector2D read FSamples2D;
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
  SetLength(FSamples1D, 0);
  SetLength(FSamples2D, 0);
  inherited;
end;

procedure TSamplesSet.LoadCSVFile(const AFileName: string; const AInputSize, AOutputSize: Word; const ADelimiter: Char);
var
  iRow, iCol, iCount: Cardinal;
  CSVFile: TStringList;
  Line: TStringList;
begin
  if not FileExists(AFileName) then
    raise Exception.CreateFmt('O arquivo %s não existe.', [AFileName]);

  if ADelimiter = '' then
    raise Exception.Create('É necessário informar um delimitador para carregar os dados no formato CSV.');

  Line := TStringList.Create;
  CSVFile := TStringList.Create;
  try
    CSVFile.LoadFromFile(AFileName);
    Line.Delimiter := ADelimiter;

    // armazena os parâmetros de entrada
    FInputSize := AInputSize;
    FOutputSize := AOutputSize;
    FSampleSize := AInputSize + AOutputSize;
    FSamplesCount := CSVFile.Count;

    // aloca memória para os dados
    // para o vetor FSamples1D, o BIAS é armazenado para facilitar o processamento na GPU
    SetLength(FSamples1D, FSamplesCount * (FSampleSize + 1)); // +1 for BIAS
    SetLength(FSamples2D, FSamplesCount, FSampleSize);

    // carrega os dados nos vetores
    iCount := 0;
    for iRow := 0 to FSamplesCount - 1 do
    begin
      Line.DelimitedText := CSVFile.Strings[iRow];

      for iCol := 0 to FSampleSize - 1 do
      begin
        FSamples2D[iRow][iCol] := StrToFloat(Line.Strings[iCol]);
        FSamples1D[iCount] := FSamples2D[iRow][iCol];
        Inc(iCount);
      end;
      FSamples1D[iCount] := 1;  // BIAS
      Inc(iCount);
    end;
  finally
    FreeAndNil(Line);
    FreeAndNil(CSVFile);
  end;
end;

end.
