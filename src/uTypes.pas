unit uTypes;

interface

type
  /// <summary>
  ///  Tipos de vetor utilizado para armazenar os dados.
  /// </summary>
  PVector1D = ^TVector1D;
  TVector1D = array of Single;

  PVector2D = ^TVector2D;
  TVector2D = array of TVector1D;

implementation

end.
