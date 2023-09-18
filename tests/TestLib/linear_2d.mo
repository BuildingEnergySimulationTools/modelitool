model linear_2d
  Modelica.Blocks.Sources.Constant x(k = 2)  annotation(
    Placement(visible = true, transformation(origin = {-56, 46}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.Constant y(k = 2)  annotation(
    Placement(visible = true, transformation(origin = {-56, -32}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Interaction.Show.RealValue res annotation(
    Placement(visible = true, transformation(origin = {44, 2}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
equation
res.numberPort = 4*x.k+2*y.k-6;

annotation(
    uses(Modelica(version = "4.0.0")));
end linear_2d;
