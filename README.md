# New York City Food Inspection Forecasting
## Nov, 2018

**Problem:**<br/>
Predict the NYC restaurants which have higher risk to fail the food inspection.<br/>

**Target:**<br/>
Yes/No - having multiple critical violations

**Datasets:**<br/>
1) NYC Open Data - DOHMH New York City Restaurant Inspection Results:<br/>
This is our core data containing the target variable, which contains all violation citations from the Department of Health and Mental Hygiene (DOHMH) inspection conducted up to three years prior to the most recent inspection for restaurants in New York City.<br/>
2) Yelp Fusion API:<br/>
The Yelp Fusion API allows us to get the restaurant profile and user reviews from businesses of interest.<br/>
3) Google Places API:<br/>
This API allows us to request user reviews about the indicated restaurant.<br/>
4) NOAA - Local Climatological Data (LCD):<br/>
We get New York City hourly weather data from this online data tools. This contains temperature and humidity data that is relevant to our problem, since fresh food, thawed foods are sensitive to temperatures. Improper temperature control for food, are most susceptible to accommodating the start or spread of food borne illnesses.<br/>
5) NYC Open Data - Rodent Complaints in NYC:<br/>
It is fair to say that the worst possible scenario for any restaurant is a rodent problem. We use this dataset to identify which area’s restaurants are more likely to suffer from rodent infestation, therefore scoring lower in inspection.<br/>
6) NYU Furman Center - New York City Neighborhood Data Profiles:<br/>
   The William and Anita Newman Library - NYC Geographies:<br/>
Neighborhood data is critical for understanding local demographic and identifying community needs. We are interested to know whether neighborhood profiles would have a influence on restaurants’ performances in inspection.<br/>

**Main Models:**<br/>
1) Logistic Regression<br/>
2) Random Forest<br/>
3) LightGBM (AUC: 0.7438)<br/>
4) Ensemble<br/>
Notes: also tried Keras 2-layer Neural Network and non-linear SVM, can't beat the LightGBM. Details not included in the final report.<br/>
