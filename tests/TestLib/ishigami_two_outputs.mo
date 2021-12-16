model ishigami_two_outputs
  Modelica.Blocks.Sources.Constant x(k = 2);
  Modelica.Blocks.Sources.Constant y(k = 2);
  Modelica.Blocks.Sources.Constant z(k = 2);
  
  Real A1=7.0;
  Real B1=0.1;
  
  Real A2=5.0;
  Real B2=0.5;
  
  Modelica.Blocks.Interaction.Show.RealValue res1 annotation(
    Placement(visible = true, transformation(origin = {44, 2}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Interaction.Show.RealValue res2 annotation(
    Placement(visible = true, transformation(origin = {44, -22}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
equation
  res1.numberPort = Modelica.Math.sin(x.k)+A1*Modelica.Math.sin(y.k)^2+B1*z.k^4*Modelica.Math.sin(x.k);
  res2.numberPort = Modelica.Math.sin(x.k)+A2*Modelica.Math.sin(y.k)^2+B2*z.k^4*Modelica.Math.sin(x.k);

annotation(
    uses(Modelica(version = "3.2.3")));
end ishigami_two_outputs;
