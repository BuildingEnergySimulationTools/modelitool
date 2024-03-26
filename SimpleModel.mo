model SimpleModel
  parameter Test_library.Data.params_pumpconsumption params annotation(
    Placement(visible = true, transformation(origin = {0, 2}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.RealExpression realExpression(y = params.c)  annotation(
    Placement(visible = true, transformation(origin = {-4, 46}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Blocks.Interfaces.RealOutput y annotation(
    Placement(visible = true, transformation(origin = {76, 16}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {76, 16}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
equation
  connect(realExpression.y, y) annotation(
    Line(points = {{8, 46}, {30, 46}, {30, 16}, {76, 16}}, color = {0, 0, 127}));

annotation(
    uses(Modelica(version = "4.0.0"), Test_library(version="default")));
end SimpleModel;
