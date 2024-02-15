model BinhandKorn
  // BNH test from pymoo
  Modelica.Blocks.Sources.Constant x(k = 2);
  Modelica.Blocks.Sources.Constant y(k = 2);
  // 0 <= x <= 5
  // 0 <= y <= 5
  Real f1;
  Real f2;
  Real g1;
  Real g2;
equation
  f1 = 4*x.k^2 + 4*y.k^2;
  f2 = (x.k - 5)^2 + (y.k-5)^2;
  g1 = (x.k - 5)^2 + y.k^2 - 25 ;
  g2 = 7.7 - (x.k - 8)^2 - (y.k+3)^2 ;
  
  
annotation(
    uses(Modelica(version = "4.0.0")));
end BinhandKorn;
