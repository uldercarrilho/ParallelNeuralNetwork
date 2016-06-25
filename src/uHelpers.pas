unit uHelpers;

interface

uses
  System.Classes;

type
  TStringsHelper = class helper for TStrings
  public
    /// <summary>
    ///  Adiciona uma nova linha na lista interna, aplicando a formata��o definida no par�metro Args.
    /// </summary>
    /// <param name="S">
    ///  Texto que ser� inserido na lista interna.
    /// </param>
    /// <param name="Args">
    ///  Argumentos que ser�o utilizados para formatar o texto do par�metro S. A sintaxe � a mesma utilizada pela fun��o
    ///  Format.
    /// </param>
    /// <returns>
    ///  Posi��o do item na lista.
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

