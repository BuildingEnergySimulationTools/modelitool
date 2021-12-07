model etics_v0
  parameter Real lambda_concrete = 1;
  parameter Real lambda_coating = 1;
  parameter Real capa_concrete = 1000;
  parameter Real capa_ins = 1000;
  parameter Real capa_coating = 1000;
  parameter Real rho_concrete = 875;
  parameter Real rho_ins = 40;
  parameter Real rho_coating = 200;
  parameter Real ep_concrete = 0.2;
  parameter Real ep_ins = 0.2;
  parameter Real ep_coating = 0.01;
  Modelica.Blocks.Sources.Constant Alpha_clo(k = 0.5);
  Modelica.Blocks.Sources.Constant R_conv_ext(k = 0.04);
  Modelica.Blocks.Sources.Constant R_conv_int(k = 0.13);
  Modelica.Blocks.Sources.Constant Lambda_ins(k = 0.04);
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor R_w_in(R = ep_concrete / (lambda_concrete * 2)) annotation(
    Placement(visible = true, transformation(origin = {70, -10}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor R_w_out(R = ep_concrete / (lambda_concrete * 2)) annotation(
    Placement(visible = true, transformation(origin = {30, -10}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor C_w(C = ep_concrete * rho_concrete * capa_concrete, T(displayUnit = "K", start = 273.15 + 25)) annotation(
    Placement(visible = true, transformation(origin = {50, -30}, extent = {{-10, 10}, {10, -10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor R_ins1_out(R = ep_ins / (Lambda_ins.k * 2)) annotation(
    Placement(visible = true, transformation(origin = {-50, -10}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor C_ins1(C = ep_ins * rho_ins * capa_ins) annotation(
    Placement(visible = true, transformation(origin = {-30, -30}, extent = {{-10, 10}, {10, -10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor R_ins1_in(R = ep_ins / (Lambda_ins.k * 2)) annotation(
    Placement(visible = true, transformation(origin = {-10, -10}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor R_ins2_in(R = ep_ins / (Lambda_ins.k * 2)) annotation(
    Placement(visible = true, transformation(origin = {-90, -10}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor C_ins2(C = ep_ins * rho_ins * capa_ins) annotation(
    Placement(visible = true, transformation(origin = {-110, -30}, extent = {{-10, 10}, {10, -10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor R_ins2_out(R = ep_ins / (Lambda_ins.k * 2)) annotation(
    Placement(visible = true, transformation(origin = {-130, -10}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor R_c_in(R = ep_coating / (lambda_coating * 2)) annotation(
    Placement(visible = true, transformation(origin = {-170, -10}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor R_c_out(R = ep_coating / (lambda_coating * 2)) annotation(
    Placement(visible = true, transformation(origin = {-210, -10}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor C_c(C = ep_coating * rho_coating * capa_coating) annotation(
    Placement(visible = true, transformation(origin = {-190, -30}, extent = {{-10, 10}, {10, -10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor Fictive_in(C = 42, T(displayUnit = "K")) annotation(
    Placement(visible = true, transformation(origin = {130, -30}, extent = {{-10, 10}, {10, -10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedTemperature T_sky annotation(
    Placement(visible = true, transformation(origin = {-310, 50}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Components.BodyRadiation IR_sky(Gr = 0.4) annotation(
    Placement(visible = true, transformation(origin = {-250, 50}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Components.ConvectiveResistor Conv_ext annotation(
    Placement(visible = true, transformation(origin = {-250, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Components.BodyRadiation IR_Amb(Gr = 0.6) annotation(
    Placement(visible = true, transformation(origin = {-250, 10}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedTemperature T_ext annotation(
    Placement(visible = true, transformation(origin = {-310, -30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow Solar_rad annotation(
    Placement(visible = true, transformation(origin = {-250, -70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Components.ConvectiveResistor Conv_in annotation(
    Placement(visible = true, transformation(origin = {110, -10}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedTemperature T_in annotation(
    Placement(visible = true, transformation(origin = {150, -10}, extent = {{10, -10}, {-10, 10}}, rotation = 0)));
  Modelica.Blocks.Sources.CombiTimeTable Boundaries(columns = 2:7, fileName = "C:/Users/bdurandestebe/Documents/45_MODELITOOL/Tutorial/boundaries.txt", tableName = "table1", tableOnFile = true) annotation(
    Placement(visible = true, transformation(origin = {-170, 90}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Celsius.ToKelvin Kelvin_Ext annotation(
    Placement(visible = true, transformation(origin = {-130, 110}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Celsius.ToKelvin Kelvin_Wall_Ins annotation(
    Placement(visible = true, transformation(origin = {-130, 130}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Celsius.ToKelvin Kelvin_Ins_Ins annotation(
    Placement(visible = true, transformation(origin = {-130, 150}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Celsius.ToKelvin Kelvin_Ins_Coat annotation(
    Placement(visible = true, transformation(origin = {-130, 170}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Celsius.ToKelvin Kelvin_Int annotation(
    Placement(visible = true, transformation(origin = {-130, 192}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Celsius.ToKelvin Kelvin_sky annotation(
    Placement(visible = true, transformation(origin = {-350, 50}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Sensors.TemperatureSensor Tw_out annotation(
    Placement(visible = true, transformation(origin = {30, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Sensors.TemperatureSensor T_ins_ins annotation(
    Placement(visible = true, transformation(origin = {-50, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Sensors.TemperatureSensor T_coat_ins annotation(
    Placement(visible = true, transformation(origin = {-130, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
equation
  Solar_rad.Q_flow = Boundaries.y[2] * Alpha_clo.k;
  Kelvin_sky.Kelvin = 0.0552 * Kelvin_Ext.Kelvin^1.5;
  connect(Kelvin_Ext.Kelvin, T_ext.T);
  connect(Kelvin_Int.Kelvin, T_in.T);
  connect(Conv_ext.Rc, R_conv_ext.y);
  connect(Conv_in.Rc, R_conv_int.y);
  connect(R_w_in.port_a, R_w_out.port_b) annotation(
    Line(points = {{60, -10}, {40, -10}}, color = {191, 0, 0}));
  connect(R_w_out.port_a, R_ins1_in.port_b) annotation(
    Line(points = {{20, -10}, {0, -10}}, color = {191, 0, 0}));
  connect(R_ins1_in.port_a, R_ins1_out.port_b) annotation(
    Line(points = {{-20, -10}, {-40, -10}}, color = {191, 0, 0}));
  connect(R_ins1_out.port_a, R_ins2_in.port_b) annotation(
    Line(points = {{-60, -10}, {-80, -10}}, color = {191, 0, 0}));
  connect(R_ins2_in.port_a, R_ins2_out.port_b) annotation(
    Line(points = {{-100, -10}, {-120, -10}}, color = {191, 0, 0}));
  connect(R_ins2_out.port_a, R_c_in.port_b) annotation(
    Line(points = {{-140, -10}, {-160, -10}}, color = {191, 0, 0}));
  connect(R_c_in.port_a, R_c_out.port_b) annotation(
    Line(points = {{-180, -10}, {-200, -10}}, color = {191, 0, 0}));
  connect(R_c_out.port_b, C_c.port) annotation(
    Line(points = {{-200, -10}, {-190, -10}, {-190, -20}}, color = {191, 0, 0}));
  connect(R_ins2_out.port_b, C_ins2.port) annotation(
    Line(points = {{-120, -10}, {-110, -10}, {-110, -20}}, color = {191, 0, 0}));
  connect(R_ins1_out.port_b, C_ins1.port) annotation(
    Line(points = {{-40, -10}, {-30, -10}, {-30, -20}}, color = {191, 0, 0}));
  connect(R_w_out.port_b, C_w.port) annotation(
    Line(points = {{40, -10}, {50, -10}, {50, -20}}, color = {191, 0, 0}));
  connect(T_ext.port, Conv_ext.solid) annotation(
    Line(points = {{-300, -30}, {-260, -30}}, color = {191, 0, 0}));
  connect(Conv_ext.fluid, R_c_out.port_a) annotation(
    Line(points = {{-240, -30}, {-230, -30}, {-230, -10}, {-220, -10}}, color = {191, 0, 0}));
  connect(IR_Amb.port_b, R_c_out.port_a) annotation(
    Line(points = {{-240, 10}, {-230, 10}, {-230, -10}, {-220, -10}}, color = {191, 0, 0}));
  connect(IR_sky.port_b, R_c_out.port_a) annotation(
    Line(points = {{-240, 50}, {-230, 50}, {-230, -10}, {-220, -10}}, color = {191, 0, 0}));
  connect(T_sky.port, IR_sky.port_a) annotation(
    Line(points = {{-300, 50}, {-260, 50}}, color = {191, 0, 0}));
  connect(T_ext.port, IR_Amb.port_a) annotation(
    Line(points = {{-300, -30}, {-288, -30}, {-288, 10}, {-260, 10}}, color = {191, 0, 0}));
  connect(Solar_rad.port, R_c_out.port_a) annotation(
    Line(points = {{-240, -70}, {-230, -70}, {-230, -10}, {-220, -10}}, color = {191, 0, 0}));
  connect(R_w_in.port_b, Conv_in.solid) annotation(
    Line(points = {{80, -10}, {100, -10}}, color = {191, 0, 0}));
  connect(Conv_in.fluid, Fictive_in.port) annotation(
    Line(points = {{120, -10}, {130, -10}, {130, -20}}, color = {191, 0, 0}));
  connect(T_in.port, Fictive_in.port) annotation(
    Line(points = {{140, -10}, {130, -10}, {130, -20}}, color = {191, 0, 0}));
  connect(Boundaries.y[1], Kelvin_Ext.Celsius) annotation(
    Line(points = {{-159, 90}, {-153, 90}, {-153, 110}, {-143, 110}}, color = {0, 0, 127}));
  connect(Boundaries.y[3], Kelvin_Wall_Ins.Celsius) annotation(
    Line(points = {{-159, 90}, {-153, 90}, {-153, 130}, {-143, 130}}, color = {0, 0, 127}));
  connect(Boundaries.y[4], Kelvin_Ins_Ins.Celsius) annotation(
    Line(points = {{-159, 90}, {-153, 90}, {-153, 150}, {-143, 150}}, color = {0, 0, 127}));
  connect(Boundaries.y[5], Kelvin_Ins_Coat.Celsius) annotation(
    Line(points = {{-159, 90}, {-153, 90}, {-153, 170}, {-143, 170}}, color = {0, 0, 127}));
  connect(Boundaries.y[6], Kelvin_Int.Celsius) annotation(
    Line(points = {{-159, 90}, {-153, 90}, {-153, 192}, {-143, 192}}, color = {0, 0, 127}));
  connect(Kelvin_sky.Kelvin, T_sky.T) annotation(
    Line(points = {{-338, 50}, {-322, 50}}, color = {0, 0, 127}));
  connect(R_ins1_in.port_b, Tw_out.port) annotation(
    Line(points = {{0, -10}, {8, -10}, {8, 30}, {20, 30}}, color = {191, 0, 0}));
  connect(R_ins2_in.port_b, T_ins_ins.port) annotation(
    Line(points = {{-80, -10}, {-70, -10}, {-70, 30}, {-60, 30}}, color = {191, 0, 0}));
  connect(R_c_in.port_b, T_coat_ins.port) annotation(
    Line(points = {{-160, -10}, {-150, -10}, {-150, 30}, {-140, 30}}, color = {191, 0, 0}));
  annotation(
    uses(Modelica(version = "3.2.3")),
    Diagram(coordinateSystem(extent = {{-360, 160}, {180, -80}})),
    version = "",
    experiment(StartTime = 6.912e+06, StopTime = 7.0845e+06, Tolerance = 1e-06, Interval = 300),
    __OpenModelica_commandLineOptions = "--matchingAlgorithm=PFPlusExt --indexReductionMethod=dynamicStateSelection -d=initialization,NLSanalyticJacobian",
    __OpenModelica_simulationFlags(lv = "LOG_STATS", s = "dassl"));
end etics_v0;
