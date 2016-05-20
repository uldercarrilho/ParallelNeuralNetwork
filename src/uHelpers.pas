unit uHelpers;

interface

uses
  System.Classes;

type
  TStringsHelper = class helper for TStrings
  public
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
