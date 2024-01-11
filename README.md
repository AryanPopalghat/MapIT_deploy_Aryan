# Model-Deployment-MarketGPT-Project

Command to start -->
uvicorn app:app --host 0.0.0.0 --port 10000

Model is deployed on Render
https://model-deployment-marketgpt-project-1.onrender.com

All code files have been added in this git repo
https://github.com/ManasiBharatIngle/Model-Deployment-MarketGPT-Project

Currently the model will be taking 10 sales values, each corresponding to one lag as input and will be giving the scaled sales prediction as output.
To get the actual sales value we will have to change the model pipeline. Other features can also be added according to our requirement.

After https://model-deployment-marketgpt-project-1.onrender.com this, add "/docs" in the url ( "https://model-deployment-marketgpt-project-1.onrender.com/docs" ) to get the FastAPI page for sales prediction.
