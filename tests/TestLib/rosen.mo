model rosen
  Modelica.Blocks.Sources.Constant x(k = 2)  annotation(
    Placement(visible = true, transformation(origin = {-56, 46}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.Constant y(k = 2)  annotation(
    Placement(visible = true, transformation(origin = {-56, -32}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Interaction.Show.RealValue res annotation(
    Placement(visible = true, transformation(origin = {44, 2}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
equation
res.numberPort = (1-x.k)^2+100*(y.k-x.k^2)^2;

annotation(
    uses(Modelica(version = "3.2.3")));
end rosen;
