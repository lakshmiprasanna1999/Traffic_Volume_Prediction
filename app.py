from flask import *
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

app=Flask(__name__)

@app.route("/")
@app.route("/home",methods=["GET","POST"])
def home():
	if request.method=="POST":
		
		data = pd.read_csv(r"C:\Users\Admin\Desktop\Train.csv")

		data["year"] = data["date_time"].apply(lambda x : x.split(" ")[0].split("-")[0])
		data["month"] = data["date_time"].apply(lambda x : x.split(" ")[0].split("-")[1])
		data["day"] = data["date_time"].apply(lambda x : x.split(" ")[0].split("-")[2])
		data["time"] = data["date_time"].apply(lambda x : x.split(" ")[1].split(":")[0])
		data["day_of_week"] = pd.DatetimeIndex(data["date_time"].apply(lambda x : x.split(" ")[0])).dayofweek
		data["snow_p_h"] = data["snow_p_h"].apply(lambda x : 1 if x!=0 else 0)          
		data["wind_direction"] = data["wind_direction"].apply(lambda x : x//90)
		data["wind_direction"] = data["wind_direction"].apply(lambda x : 0 if x == 4 else x)
		data["speed_temp"] = np.sqrt(np.multiply(data["wind_speed"],data["temperature"]))
		time_arr = data["time"].values
		c=0
		for t in range(len(time_arr)-1):
			if int(time_arr[t+1])==int(time_arr[t]):
				c+=1
		# taking holiday as a feature

		for i in range(len(data)):
			if data.at[i,"is_holiday"] != "None":
				d = str(data.at[i,"date_time"].split(" ")[0])
				j=i
				while str(data.at[j,"date_time"].split(" ")[0])==d:
					data.at[j,"is_holiday"] = 1
					j+=1
			else:
				data.at[i,"is_holiday"] = 0
		#adding dummy values
		data = pd.get_dummies(data, columns = ["day_of_week", "month", "wind_direction"], prefix_sep='_', drop_first=True)
		data = data.drop_duplicates(subset=['date_time', 'traffic_volume'], keep="last")
		mod_data = data.drop(columns=["date_time","traffic_volume","weather_description","weather_type","dew_point","visibility_in_miles"])


		X = mod_data.values
		Y = data["traffic_volume"].values

		
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.8, random_state=0)

		
		model = XGBRegressor(
                        gamma=5, 
                        learning_rate=.3,
                        max_depth=15,
                        reg_lambda=100,
                        n_estimators = 100
                        )
                         
		model.fit(X_train, Y_train)#,eval_metric='rmse', verbose = True, eval_set = [(X_test, Y_test)])

		y_pred = model.predict(X_test)

		M_error=np.sqrt(mean_squared_error(Y_test, y_pred))
		A_error=metrics.mean_absolute_error(Y_test,y_pred)

		#Y=model.predict([[air_pollution_index,humidity,temperature,wind_speed,wind_direction,clouds_all,Day,Month,Year,Hour,Minutes]])
		#y=list(Y[0])
		print(M_error,A_error)
		return render_template("home.html",M_error=M_error,A_error=A_error)
	else:
		Days=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
		Months=[1,2,3,4,5,6,7,8,9,10,11,12]
		Years=[2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]
		return render_template("home.html",Days=Days,Months=Months,Years=Years)
	
	
if __name__=="__main__":
	app.run(debug=True)