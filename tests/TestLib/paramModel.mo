within TestLib;

model paramModel
  parameter Real k=1;
equation

  annotation(
    uses(Modelica(version = "4.0.0"), Test_library(version="default")));
end paramModel;
