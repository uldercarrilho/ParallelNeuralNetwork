unit uHelpers;

interface

uses
  System.Classes;

type
  TStringsHelper = class helper for TStrings
  public
    /// <summary>
    ///  Adiciona uma nova linha na lista interna, aplicando a formatação definida no parâmetro Args.
    /// </summary>
    /// <param name="S">
    ///  Texto que será inserido na lista interna.
    /// </param>
    /// <param name="Args">
    ///  Argumentos que serão utilizados para formatar o texto do parâmetro S. A sintaxe é a mesma utilizada pela função
    ///  Format.
    /// </param>
    /// <returns>
    ///  Posição do item na lista.
    /// </returns>
    function AddFmt(const S: string; Args: array of const): Integer;
  end;

implementation

uses
  System.SysUtils;

{ TStringsHelper }

function TStringsHelper.AddFmt(const S: string; Args: array of const): Integer;
begin
  Result := Add(Format(S, Args));
end;

end.

