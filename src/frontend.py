from fastapi import FastAPI, HTTPException

from pydantic import computed_field, BaseModel, Field
from typing import Annotated

import pickle
import pandas as pd

app = FastAPI()

class inputSchema(BaseModel):
    pregnancies : Annotated[int, Field("the integer that describe the pregnancy level of the patient", examples=[1, 4, 8])]
    glucose : Annotated[int, Field("The glucose level of patient", examples=[116, 95, 181])]
    bloodpressure : Annotated[int, Field("The blood pressure reading of the patient", examples=[70, 68, 56])]
    skinthickness : Annotated[int, Field("Skin thickness of patient", examples=[28, 32, 36])]
    insulin : Annotated[int, Field("The insulin level of patient", examples=[0, 495, 105])]
    bmi : Annotated[float, Field("The bmi of the patient", examples=[27.4, 32.1, 30.1])]
    diabetespedigreefunction : Annotated[float, Field("The diabets pedigree function of the patient", examples=[0.204, 0.612, 0.148])]
    age : Annotated[int, Field("The age of the patient", examples=[21, 24, 60], gt=0, lt=100)]

    @computed_field
    @property
    def agegroup(self) -> int:
        if self.age <= 30:
            return 0           # age group 0 young
        elif self.age <= 60:
            return 1           # age group 1 elders
        return 2               # age group 2 old
    
    @computed_field
    @property
    def bmicategory(self) -> int:
        if self.bmi <= 18.5:
            return 0         # underweight
        elif self.bmi <= 24.9:
            return 1         # normal
        elif self.bmi <= 29.9:
            return 2         # overweight
        return 3 
    
    @computed_field
    @property
    def gttratio(self) -> float:
        ratio = (self.glucose) / (self.insulin + 1)
        return ratio
    
    @computed_field
    @property
    def isinsulinflag(self) -> int:
        if (self.insulin > 100) and (self.glucose > 120):
            return 1
        return 0
    
@app.get("/")
def main_page():
    return {"message" : "You landed on the diabetics main page"}

#	
@app.post("/predict")
def getInput(input : inputSchema):
    test_df = pd.DataFrame([{
        'Pregnancies' : input.pregnancies ,
        'Glucose' : input.glucose ,
        'BloodPressure' : input.bloodpressure ,
        'Insulin' : input.insulin ,
    	'DiabetesPedigreeFunction' : input.diabetespedigreefunction ,
        'AgeGroup' : input.agegroup ,
        'BMIGroup': input.bmicategory ,
        'GTIratio' : input.gttratio , 
        'isInsulinFlag' : input.isinsulinflag
    }])

    with open('models/model.pkl', 'rb') as f:
        trained_model = pickle.load(f)

    y_pred = trained_model.predict(test_df)
    prediction = int(y_pred[0])

    return {"Diabetics prediction : ": prediction}