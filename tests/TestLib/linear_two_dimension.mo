within TestLib;

model linear_two_dimension
Modelica.Blocks.Sources.Constant x(k=2);
Modelica.Blocks.Sources.Constant y(k=1);
Modelica.Blocks.Sources.CombiTimeTable Boundaries(columns = 2:3, tableName = "table1", tableOnFile = true);
Modelica.Blocks.Interaction.Show.RealValue res;

equation
res.numberPort = x.k*Boundaries.y[1]+y.k*Boundaries.y[2];

end linear_two_dimension;
