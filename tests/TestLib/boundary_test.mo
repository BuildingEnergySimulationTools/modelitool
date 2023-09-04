within TestLib;
model boundary_test
  Modelica.Blocks.Sources.CombiTimeTable Boundaries(columns = 2:3, tableName = "table1", tableOnFile = true)  annotation(
    Placement(visible = true, transformation(origin = {-70, 10}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
equation

annotation(
    uses(Modelica(version = "3.2.3")));
end boundary_test;
