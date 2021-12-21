within TestLib;

model linear_two_dimension
Modelica.Blocks.Sources.Constant x(k=2);
Modelica.Blocks.Sources.Constant y(k=1);
Modelica.Blocks.Sources.CombiTimeTable Boundaries(columns = 2:3, fileName = Modelica.Utilities.Files.loadResource("modelica://TestLib/ressources/linear_two_dimension_bound.txt"), tableName = "table1", tableOnFile = true);
Modelica.Blocks.Interaction.Show.RealValue res;

equation
res.numberPort = x.k*Boundaries.y[1]+y.k*Boundaries.y[2];

end linear_two_dimension;
