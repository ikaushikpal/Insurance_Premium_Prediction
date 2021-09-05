# Guide to run "Insurance Premium Prediction.py"

  

### First open your fav terminal

make sure that you are inside deployment folder on your terminal

To make sure

  

> if you are using cmd
type `cd` and hit enter it should display "\Insurance_Premium_Prediction\deployment"

  

> if you are using powershell
type `pwd` and hit enter it should display "\Insurance_Premium_Prediction\deployment"

  

> if you are using bash terminal
type `pwd` and hit enter it should display "/Insurance_Premium_Prediction/deployment"

  

if you are not inside type `cd deployment` and hit enter

  

### Need to install dependencies

  

type `pip install -U -r requirements.txt` and hit enter

  

### Start the server

type `streamlit run "Insurance Premium Prediction.py"` and hit enter

Now its working on your local computer.

### Stop the server

If you want to stop the server press `ctrl+c` on your terminal or simply close your terminal either way it will work.

___

# Info

models folder contains saved models, actually copied from "jNotebook Files/model"

"utilty.py" contains all useful functions to run this web app, such as load_data, user_input_features, scaleDF, predictTarget.

"Insurance Premium Prediction.py" containing all remaining texts and complete usage.