###The datasets use for classification### give biomass growth data, labeled based on the SDE model use for generate the dataset(standard logistic SDE or Irradiance SDE)
For 'simulated_X_data_irradiance_time_independent_noise_0.25.csv' dataset, the noise is add by the following code, which is independent from time:
	noise <- rnorm(1,0,0.25) 
  	s_irradiance <- expression(noise)# idependent NOISE  
For 'simulated_label_data_irradiance_time_dependent_noise_0.2.csv' dataset, the noise is add by the following code, which is related to biomass:
	s_irradiance <- expression(0.2*((2*(Mmax-x)/Mmax)*(1-(Mmax-x)/Mmax)))

the parameters are in ranges:
	r=rnorm(1,0.25,0.25) #22/06 update
	Max_biomass rnorm(1,6000,100)
	Y0=4.6/1000
