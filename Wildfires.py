{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "ee32845e-40d2-4784-98ba-9bf2feeb6650",
   "metadata": {},
   "source": [
    "\n",
    "[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://8dd6shpox5gp5pz.studio.us-east-2.sagemaker.aws/studiolab/default/jupyter/lab/tree/sagemaker-studiolab-notebooks/AWS-disaster-response/Wildfires.ipynb)\n",
    "\n",
    "## In this notebook we will demonstrate how to predict wildfire occurrences using machine learning based on Remote Sensing data\n",
    "\n",
    "#### The dataset comes from remote sensors is preprocessed and contains the following variables, (NDVI: Normalized Difference Vegetation Index), meteorological conditions (LST: Land Surface Temperature) as well as the fire indicator “Thermal Anomalies” (BURNED_AREA). \n",
    "\n",
    " - The NDVI is a simple graphical indicator that can be used to analyze remote sensing measurements, often from a space platform, assessing whether or not the target being observed contains live green vegetation. NDVI values are between 0 and 1, values near 0 indicate very sparse vegetation and values near 1 indicate dense vegetation.\n",
    "\n",
    "- The LST represents the radiative skin temperature of the land surface derived from solar radiation, it depends on the vegetation cover and the soil moisture\n",
    "\n",
    "- Thermal Anomalies are a fire detection strategy based on complete detection of a fire (only when the fire is strong enough to be detected.\n",
    "\n",
    "#### All three parameters were collected from MODIS (Moderate Resolution Imaging Spectroradiometer), an instrument carried on board the NASA's Terra platform. The fourth variable which will be our response is whether a fire occurred or not based on those parameters.\n",
    "\n",
    "### Reference:\n",
    "\n",
    "#### Younes Oulad Sayada, Hajar Mousannifb, Hassan Al Moatassime [Predictive Modeling of Wildfires: A New Dataset and Machine Learning Approach](https://pdf.sciencedirectassets.com/271100/1-s2.0-S0379711219X00028/1-s2.0-S0379711218303941/am.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEFMaCXVzLWVhc3QtMSJHMEUCIQCGBnF8JddFcIL77%2BM4y0cARiUTMypLs%2FExPRLAX%2BsjgQIgH2XGfPwWwiaSqLm5WkHpv5JJzwylfT3RN2fRSsVj6wUqgwQIjP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAEGgwwNTkwMDM1NDY4NjUiDAETElCpoiE0SOxkyyrXAzpzR2%2FLAjHBaDFMW4whSniOG2Hedg5RbPveXgNU1mPGhj6Oz%2FKThf5%2BsfZaz4UhA0emyZxHbRppEUpEmii4l9Pn3lPU%2FQJItFktjgvFMXBOyC%2FEF7URINF493oURcStp3cwmAIl4WZ6ZeD7%2F9TPbUJ2khW348Lcqi1iBaI6rJH9O3nHXeFFariOhRFxaraz7fFd%2B65t1CQTpR%2FTwZyoDWHdSfjoeoW4Ecpj8jwvV8TOsUxyNprZXX9XpJt1GraF9sIIDutBVWor0evxo9K4vmZ47hVXzfH5ObYQ4JNBiD2PXJqBUO9CzW2hpENqi5YC%2Fz3vGKbdNx5uzY26YHKO%2F9BR%2FcuTg5forNfvwUF1w8WCn6SMx6KCK%2FeQbxT1IsYXWexDwo7LQTVoWl6nyuPhtgl9iqmRmqqoMDKF1qDXhr4oTmhZPqq9t5Y%2FytZ2xQGLptV832s%2FSK8DJgZzOgbUJb3JAvwlhon9IFC5Nsd8Hr%2FrhanKj2a%2FAsloSVfLSaJBby6OM%2BtJMIdA%2Bv2VVWk90X7rsg2CUt6dcYsSAB6%2Bqn8boJ0JsUu0awghrdcjOQKSLR4NFj8RUYfQUIxP%2BJeXSpoW4Z8ZfcBF%2F%2Bqb%2FdINXbA8m9QIYmlxJzC9jvSPBjqlAaQ1hKa%2BQLv%2BVDFKUxArBbx9JWru7W2OjauvtJcwBl8WKrnJkgtq4WvaZl8M0q34BtUOAUGZQqxZ%2BOPvp6W8Df377ovwbhc%2FrS07r2soW2pIHX2QKV5OukZ7ANOeohquv8TkhMmYQi4XkDvnXKPxJujh6tA4ZOvyQa992HV3u9PdzNHhQ5rvcpAX7vAkKYmmO2k%2BOvE7ez7zp%2FGrftco%2BVaGZ2zRcw%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20220204T114927Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYQQVHBCQE%2F20220204%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=105f44320308b3aa81336ef300831b2c111da5fca475355d10d0594649bccf10&hash=6e5cf2719f6409013fc0ca0b735b617cb48b73635ab255e05e41c4e42b4a9525&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0379711218303941&tid=pdf-10c6c5d8-5edd-420f-9d36-0e4f70bfa8d9&sid=7e7f4a2654f7754682087e3287c2e6bb6e53gxrqb&type=client)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5bc10a33-f9de-40f8-9313-b9ccaaf16946",
   "metadata": {},
   "source": [
    "#### Import required libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "01bea2b9-cb3e-4500-affa-eb8a65720678",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pandas in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (1.4.0)\n",
      "Requirement already satisfied: python-dateutil>=2.8.1 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from pandas) (2.8.2)\n",
      "Requirement already satisfied: numpy>=1.18.5 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from pandas) (1.22.1)\n",
      "Requirement already satisfied: pytz>=2020.1 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from pandas) (2021.3)\n",
      "Requirement already satisfied: six>=1.5 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from python-dateutil>=2.8.1->pandas) (1.16.0)\n"
     ]
    }
   ],
   "source": [
    "!pip install pandas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "17923e3c-72b4-48b6-9b92-995932a72591",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: matplotlib in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (3.5.1)\n",
      "Requirement already satisfied: scikit-learn in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (1.0.2)\n",
      "Requirement already satisfied: xgboost in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (1.5.2)\n",
      "Requirement already satisfied: pyparsing>=2.2.1 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from matplotlib) (3.0.6)\n",
      "Requirement already satisfied: pillow>=6.2.0 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from matplotlib) (9.0.0)\n",
      "Requirement already satisfied: numpy>=1.17 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from matplotlib) (1.22.1)\n",
      "Requirement already satisfied: cycler>=0.10 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from matplotlib) (0.11.0)\n",
      "Requirement already satisfied: fonttools>=4.22.0 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from matplotlib) (4.29.0)\n",
      "Requirement already satisfied: python-dateutil>=2.7 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from matplotlib) (2.8.2)\n",
      "Requirement already satisfied: packaging>=20.0 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from matplotlib) (21.3)\n",
      "Requirement already satisfied: kiwisolver>=1.0.1 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from matplotlib) (1.3.2)\n",
      "Requirement already satisfied: scipy>=1.1.0 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from scikit-learn) (1.7.3)\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from scikit-learn) (3.1.0)\n",
      "Requirement already satisfied: joblib>=0.11 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from scikit-learn) (1.1.0)\n",
      "Requirement already satisfied: six>=1.5 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from python-dateutil>=2.7->matplotlib) (1.16.0)\n"
     ]
    }
   ],
   "source": [
    "!pip install matplotlib scikit-learn xgboost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "d9c2a90b-272f-4f11-9daf-0130badac88a",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages/xgboost/compat.py:36: FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.\n",
      "  from pandas import MultiIndex, Int64Index\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "\n",
    "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "import xgboost as xgb\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import cross_val_score, validation_curve, train_test_split, GridSearchCV\n",
    "from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore') \n",
    "\n",
    "from IPython.display import Image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "411c7ca2-b21a-4c86-8536-62f341a19cd8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'/home/studio-lab-user/sagemaker-studiolab-notebooks/AWS-disaster-response'"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pwd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "6253bff8-9414-496b-96f9-95a52a8b7693",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "#os.mkdir('/home/studio-lab-user/sagemaker-studiolab-notebooks/AWS-disaster-response')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "7e1726c4-9869-4c8e-b990-374e4a6411c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "#os.chdir('/home/studio-lab-user/sagemaker-studiolab-notebooks/AWS-disaster-response')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "055f0a41-c043-4f4e-b91c-0c76ba544ea8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'/home/studio-lab-user/sagemaker-studiolab-notebooks/AWS-disaster-response'"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pwd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "6f02dc9f-9484-44ed-8cee-32142f6f9b51",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: requests in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (2.27.1)\n",
      "Requirement already satisfied: idna<4,>=2.5 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from requests) (3.3)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.21.1 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from requests) (1.26.8)\n",
      "Requirement already satisfied: charset-normalizer~=2.0.0 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from requests) (2.0.11)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from requests) (2021.10.8)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install requests"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "c37ebc6b-7835-4ea6-ae84-99d065ab2b83",
   "metadata": {},
   "outputs": [],
   "source": [
    "import requests\n",
    "\n",
    "url = 'https://raw.githubusercontent.com/rapha18th/Wildfires-prediction-ignis/master/Wildfires.csv'\n",
    "res = requests.get(url, allow_redirects=True)\n",
    "with open('Wildfires.csv', 'wb') as file:\n",
    "    file.write(res.content)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "d4e6837d-f188-4aa1-ae0d-8ac2f7b81d74",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Wildfires.csv  Wildfires.ipynb  model_randomforest1.pkl\n"
     ]
    }
   ],
   "source": [
    "ls"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "76cdc882-388d-494d-9e20-997ac41af713",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(\"Wildfires.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "f15bcb75-70f2-4cd5-a8cf-41fc5f1e474e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>NDVI</th>\n",
       "      <th>LST</th>\n",
       "      <th>BURNED_AREA</th>\n",
       "      <th>CLASS</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.506782</td>\n",
       "      <td>14584.272727</td>\n",
       "      <td>4.692308</td>\n",
       "      <td>no_fire</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.522150</td>\n",
       "      <td>14655.833333</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>no_fire</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.682284</td>\n",
       "      <td>14780.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>fire</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.120046</td>\n",
       "      <td>13298.500000</td>\n",
       "      <td>3.500000</td>\n",
       "      <td>no_fire</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.568734</td>\n",
       "      <td>14743.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>no_fire</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       NDVI           LST  BURNED_AREA    CLASS\n",
       "0  0.506782  14584.272727     4.692308  no_fire\n",
       "1  0.522150  14655.833333     5.000000  no_fire\n",
       "2  0.682284  14780.000000     5.000000     fire\n",
       "3  0.120046  13298.500000     3.500000  no_fire\n",
       "4  0.568734  14743.000000     5.000000  no_fire"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "0b9c34f2-8b39-4c4d-b059-cfdde7ea5a19",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1713, 4)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "23c18c80-e081-46af-a6e7-2b17c46b7c64",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>NDVI</th>\n",
       "      <th>LST</th>\n",
       "      <th>BURNED_AREA</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1713.000000</td>\n",
       "      <td>1713.000000</td>\n",
       "      <td>1713.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>0.555665</td>\n",
       "      <td>14622.802073</td>\n",
       "      <td>4.674973</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.092847</td>\n",
       "      <td>366.927861</td>\n",
       "      <td>0.583791</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.030735</td>\n",
       "      <td>13137.000000</td>\n",
       "      <td>3.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>0.501276</td>\n",
       "      <td>14406.000000</td>\n",
       "      <td>4.521951</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>0.565181</td>\n",
       "      <td>14645.750000</td>\n",
       "      <td>4.920635</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>0.620987</td>\n",
       "      <td>14881.300000</td>\n",
       "      <td>5.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>0.781723</td>\n",
       "      <td>15611.570513</td>\n",
       "      <td>9.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              NDVI           LST  BURNED_AREA\n",
       "count  1713.000000   1713.000000  1713.000000\n",
       "mean      0.555665  14622.802073     4.674973\n",
       "std       0.092847    366.927861     0.583791\n",
       "min       0.030735  13137.000000     3.000000\n",
       "25%       0.501276  14406.000000     4.521951\n",
       "50%       0.565181  14645.750000     4.920635\n",
       "75%       0.620987  14881.300000     5.000000\n",
       "max       0.781723  15611.570513     9.000000"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ee2481a3-8daf-4906-8413-6153064234d0",
   "metadata": {},
   "source": [
    "#### Convert response variable from categorical to numeric"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "30106107-a305-41a5-8c4b-b40a05a4a529",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['CLASS'].replace({\"no_fire\":0, \"fire\":1}, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "2bc62baa-2410-48cf-b78c-d3b4b2340886",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>NDVI</th>\n",
       "      <th>LST</th>\n",
       "      <th>BURNED_AREA</th>\n",
       "      <th>CLASS</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1713.000000</td>\n",
       "      <td>1713.000000</td>\n",
       "      <td>1713.000000</td>\n",
       "      <td>1713.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>0.555665</td>\n",
       "      <td>14622.802073</td>\n",
       "      <td>4.674973</td>\n",
       "      <td>0.225336</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.092847</td>\n",
       "      <td>366.927861</td>\n",
       "      <td>0.583791</td>\n",
       "      <td>0.417925</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.030735</td>\n",
       "      <td>13137.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>0.501276</td>\n",
       "      <td>14406.000000</td>\n",
       "      <td>4.521951</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>0.565181</td>\n",
       "      <td>14645.750000</td>\n",
       "      <td>4.920635</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>0.620987</td>\n",
       "      <td>14881.300000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>0.781723</td>\n",
       "      <td>15611.570513</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              NDVI           LST  BURNED_AREA        CLASS\n",
       "count  1713.000000   1713.000000  1713.000000  1713.000000\n",
       "mean      0.555665  14622.802073     4.674973     0.225336\n",
       "std       0.092847    366.927861     0.583791     0.417925\n",
       "min       0.030735  13137.000000     3.000000     0.000000\n",
       "25%       0.501276  14406.000000     4.521951     0.000000\n",
       "50%       0.565181  14645.750000     4.920635     0.000000\n",
       "75%       0.620987  14881.300000     5.000000     0.000000\n",
       "max       0.781723  15611.570513     9.000000     1.000000"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cbd91259-b7de-4a03-a8b5-823518cab847",
   "metadata": {},
   "source": [
    "### Visualise distribution of variables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "46d37b35-9b5e-4a9f-9994-d8b7589f3bcd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting seaborn\n",
      "  Downloading seaborn-0.11.2-py3-none-any.whl (292 kB)\n",
      "\u001b[K     |████████████████████████████████| 292 kB 2.9 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: matplotlib>=2.2 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from seaborn) (3.5.1)\n",
      "Requirement already satisfied: scipy>=1.0 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from seaborn) (1.7.3)\n",
      "Requirement already satisfied: pandas>=0.23 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from seaborn) (1.4.0)\n",
      "Requirement already satisfied: numpy>=1.15 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from seaborn) (1.22.1)\n",
      "Requirement already satisfied: kiwisolver>=1.0.1 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from matplotlib>=2.2->seaborn) (1.3.2)\n",
      "Requirement already satisfied: pyparsing>=2.2.1 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from matplotlib>=2.2->seaborn) (3.0.6)\n",
      "Requirement already satisfied: packaging>=20.0 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from matplotlib>=2.2->seaborn) (21.3)\n",
      "Requirement already satisfied: pillow>=6.2.0 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from matplotlib>=2.2->seaborn) (9.0.0)\n",
      "Requirement already satisfied: python-dateutil>=2.7 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from matplotlib>=2.2->seaborn) (2.8.2)\n",
      "Requirement already satisfied: fonttools>=4.22.0 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from matplotlib>=2.2->seaborn) (4.29.0)\n",
      "Requirement already satisfied: cycler>=0.10 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from matplotlib>=2.2->seaborn) (0.11.0)\n",
      "Requirement already satisfied: pytz>=2020.1 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from pandas>=0.23->seaborn) (2021.3)\n",
      "Requirement already satisfied: six>=1.5 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from python-dateutil>=2.7->matplotlib>=2.2->seaborn) (1.16.0)\n",
      "Installing collected packages: seaborn\n",
      "Successfully installed seaborn-0.11.2\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install seaborn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "d4218be7-8274-4cd7-bbd9-787bc1392fc3",
   "metadata": {},
   "outputs": [],
   "source": [
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "69b87909-6c67-43c1-9d26-01f7432e5a92",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x7f76ac0d9310>"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.displot(df, x=\"NDVI\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "5d8d892f-c02c-4b63-a64e-f9f1daa65be0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x7f767574dcd0>"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.displot(df, x=\"LST\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "31eb74dd-3c46-4d02-b819-47293aabb80b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x7f767568aa90>"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.displot(df, x=\"BURNED_AREA\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "0539c83c-ce98-453a-8652-f10912c4d0d2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x7f76757b7bb0>"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.displot(df, x=\"CLASS\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "913fa178-dcf0-4840-baeb-585e1b0eccc3",
   "metadata": {},
   "source": [
    "## Build and Train Machine learning models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "0bd13f03-1ddd-42b1-88f1-24a6f9054855",
   "metadata": {},
   "outputs": [],
   "source": [
    "#features\n",
    "X = df[[\"NDVI\", \"LST\", \"BURNED_AREA\"]]\n",
    "\n",
    "#response\n",
    "y = df[\"CLASS\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "8e0f61ca-5203-4976-a58e-93f90c68ae2f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train Test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6c421128-2d89-408c-bbd2-aa3ef23c6246",
   "metadata": {},
   "source": [
    "### Model 1: Random Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 274,
   "id": "ecb9c306-5d08-472b-88b3-f5820bb1c804",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(random_state=42)"
      ]
     },
     "execution_count": 274,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "rf_model=RandomForestClassifier(random_state=42)\n",
    "\n",
    "rf_model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "cc5973f2-e928-4712-8177-11983c48067c",
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_rf = rf_model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "ad112fd0-b0bb-412a-8ee9-2416e68b3c22",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy for baseline random forest: 0.8424124513618677\n"
     ]
    }
   ],
   "source": [
    "print(\"Accuracy for baseline random forest:\", accuracy_score(y_test, pred_rf))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "8c783c43-0c16-4eb7-aaa7-403a741352b5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "F1 score for baseline randomforest: 0.547486033519553\n"
     ]
    }
   ],
   "source": [
    "print(\"F1 score for baseline randomforest:\", f1_score(y_test, pred_rf))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 304,
   "id": "79acaabf-33e0-4903-9728-6cc34679a74c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix: \n",
      "          No Fire  Fire\n",
      "No Fire      384    18\n",
      "Fire          63    49\n"
     ]
    }
   ],
   "source": [
    "print(\"Confusion Matrix: \\n\", pd.DataFrame(confusion_matrix(y_test, pred_rf),columns=['No Fire','Fire'],index=['No Fire','Fire']))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "18d378d1-ff25-4998-9d3d-9021f1f24d13",
   "metadata": {},
   "source": [
    "#### Hyperparameter Tuning for randomforest classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "9ce6ac3e-2352-42c2-993c-5c5ee19c6ca2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 5 folds for each of 120 candidates, totalling 600 fits\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "GridSearchCV(cv=5, estimator=RandomForestClassifier(random_state=42), n_jobs=1,\n",
       "             param_grid=[{'class_weight': ['balanced'],\n",
       "                          'criterion': ['gini', 'entropy'],\n",
       "                          'max_depth': [10, 50, None], 'max_features': [1, 3],\n",
       "                          'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90,\n",
       "                                           100]}],\n",
       "             refit='true', scoring='f1', verbose=1)"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "param_grid = [\n",
    "{'n_estimators':list(range(0, 110, 10))[1:], 'max_features': [1, 3], \n",
    " 'max_depth': [10, 50, None], 'criterion':['gini', 'entropy'], 'class_weight':['balanced']}\n",
    "]\n",
    "\n",
    "grid_search_forest = GridSearchCV(rf_model, param_grid, cv=5, scoring='f1', refit=\"true\",\n",
    "                                  verbose=1, n_jobs=1)\n",
    "grid_search_forest.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "c428001e-dedd-459d-b53c-6c12d453ab77",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(class_weight='balanced', max_depth=10, max_features=1,\n",
       "                       random_state=42)"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grid_search_forest.best_estimator_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9f2a1b83-71f7-49a8-b302-7200882ceacd",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "3ca61b03-f325-4fe3-bbdb-89dc9e18cffd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(class_weight='balanced', max_depth=10, max_features=1,\n",
       "                       random_state=42)"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rf_model1=RandomForestClassifier(class_weight='balanced', max_depth=10, max_features=1,\n",
    "                       random_state=42)\n",
    "\n",
    "rf_model1.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "3a763e1c-b128-4b7a-977c-a4c513980eff",
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_rf1 = rf_model1.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "b8ee112e-5b63-41e5-8aee-2ffd2b7b7309",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy for tuned random forest: 0.8268482490272373\n"
     ]
    }
   ],
   "source": [
    "print(\"Accuracy for tuned random forest:\", accuracy_score(y_test, pred_rf1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "b4b24e8b-fcea-443e-a586-418a95c8ed6f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "F1 score for tuned randomforest: 0.5741626794258373\n"
     ]
    }
   ],
   "source": [
    "print(\"F1 score for tuned randomforest:\", f1_score(y_test, pred_rf1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 305,
   "id": "96d9a43a-998e-4fe7-bc57-a49d10e7e198",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix: \n",
      "          No Fire  Fire\n",
      "No Fire      365    37\n",
      "Fire          52    60\n"
     ]
    }
   ],
   "source": [
    "print(\"Confusion Matrix: \\n\", pd.DataFrame(confusion_matrix(y_test, pred_rf1),columns=['No Fire','Fire'],index=['No Fire','Fire']))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5c52e5bd-eadd-490b-bac9-f31d05d87dc6",
   "metadata": {},
   "source": [
    "#### Since there is a heavy imbalance in the response variable, 1327 instances of “no_fire” and 386 instances of the “fire”, the appropriate evaluation metric will be F1-score. \n",
    "\n",
    "#### It is apt because the minority class is at risk for a higher proportion of false positives and false negatives due to less observations compared to the majority class. A higher F1-score means higher precision and recall (less false positives and false negatives)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "23a3c02c-384d-4b27-8ba9-65fc9444eb95",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Export the model as serialised object for use in apps"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "bc57aa7c-f582-4910-a788-999b3f8ab4b1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['model_randomforest1.pkl']"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import joblib\n",
    " \n",
    "joblib.dump(rf_model1, 'model_randomforest1.pkl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "45099f29-5c67-4413-abdc-88bb4a7879fc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Wildfires.csv  Wildfires.ipynb  model_randomforest1.pkl\n"
     ]
    }
   ],
   "source": [
    "ls"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1431054d-d287-4830-94d1-4fcce749f678",
   "metadata": {},
   "source": [
    "### Model 2: XGBoost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c739b8f0-05dc-4a32-8f57-e6b8cec4d536",
   "metadata": {},
   "outputs": [],
   "source": [
    "xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "ceecdaca-2c7b-48c2-89d3-8a1341a06b3c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages/xgboost/sklearn.py:1224: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].\n",
      "  warnings.warn(label_encoder_deprecation_msg, UserWarning)\n",
      "/home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages/xgboost/data.py:262: FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.\n",
      "  elif isinstance(data.columns, (pd.Int64Index, pd.RangeIndex)):\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[07:24:50] WARNING: ../src/learner.cc:1115: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,\n",
       "              colsample_bynode=1, colsample_bytree=1, enable_categorical=False,\n",
       "              gamma=0, gpu_id=-1, importance_type=None,\n",
       "              interaction_constraints='', learning_rate=0.300000012,\n",
       "              max_delta_step=0, max_depth=6, min_child_weight=1, missing=nan,\n",
       "              monotone_constraints='()', n_estimators=100, n_jobs=4,\n",
       "              num_parallel_tree=1, predictor='auto', random_state=42,\n",
       "              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,\n",
       "              tree_method='exact', validate_parameters=1, verbosity=None)"
      ]
     },
     "execution_count": 72,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "xgb_model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cbd15e96-1ef5-4cd9-8ee4-0abd5112db0c",
   "metadata": {},
   "source": [
    "#### Model Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "f8fbb0c9-98c0-44e2-b50c-54c0b1de738c",
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_xgb = xgb_model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "d278a222-173b-4813-8bd2-9ada7a948ba0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy for baseline XGBoost: 0.8346303501945526\n"
     ]
    }
   ],
   "source": [
    "print(\"Accuracy for baseline XGBoost:\", accuracy_score(y_test, pred_xgb))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "7b9d9cd2-c707-481d-8b7a-ea6436db3b1c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "F1 score for baseline XGBoost: 0.5685279187817259\n"
     ]
    }
   ],
   "source": [
    "print(\"F1 score for baseline XGBoost:\", f1_score(y_test, pred_xgb))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 307,
   "id": "ca1504ed-3ad4-4d47-b2ce-412838374a2c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix: \n",
      "          No Fire  Fire\n",
      "No Fire      373    29\n",
      "Fire          56    56\n"
     ]
    }
   ],
   "source": [
    "print(\"Confusion Matrix: \\n\", pd.DataFrame(confusion_matrix(y_test, pred_xgb),columns=['No Fire','Fire'],index=['No Fire','Fire']))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e0a0fef7-69e5-4d08-ba20-6d5d7aaaac22",
   "metadata": {},
   "source": [
    "#### Hyperparameter tuning for XGBoost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 310,
   "id": "e6e4d29c-948a-46fd-bd24-83038e9d26fa",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 5 folds for each of 120 candidates, totalling 600 fits\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "GridSearchCV(cv=5,\n",
       "             estimator=XGBClassifier(base_score=None, booster=None,\n",
       "                                     colsample_bylevel=None,\n",
       "                                     colsample_bynode=None,\n",
       "                                     colsample_bytree=None,\n",
       "                                     enable_categorical=False, gamma=None,\n",
       "                                     gpu_id=None, importance_type=None,\n",
       "                                     interaction_constraints=None,\n",
       "                                     learning_rate=None, max_delta_step=None,\n",
       "                                     max_depth=None, min_child_weight=None,\n",
       "                                     missing=nan, monotone_constraints=None,...\n",
       "                                     reg_lambda=None, scale_pos_weight=None,\n",
       "                                     silent=True, subsample=None,\n",
       "                                     tree_method=None, use_label_encoder=False,\n",
       "                                     validate_parameters=None, verbosity=0),\n",
       "             n_jobs=1,\n",
       "             param_grid=[{'class_weight': ['balanced'],\n",
       "                          'criterion': ['gini', 'entropy'],\n",
       "                          'max_depth': [10, 50, None], 'max_features': [1, 3],\n",
       "                          'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90,\n",
       "                                           100]}],\n",
       "             refit='true', scoring='f1', verbose=1)"
      ]
     },
     "execution_count": 310,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "xgb_1 = xgb.XGBClassifier(random_state=42, silent=True, verbosity =0, use_label_encoder=False)\n",
    "\n",
    "params_grid = { 'max_depth': [3,6,10],\n",
    "           'learning_rate': [0.01, 0.05, 0.1],\n",
    "           'n_estimators': [10, 50, 100],\n",
    "           'colsample_bytree': [0.3, 0.7]}\n",
    "\n",
    "grid_search_xgb = GridSearchCV(xgb_1, param_grid, cv=5, scoring='f1', refit=\"true\",\n",
    "                                  verbose=1, n_jobs=1)\n",
    "grid_search_xgb.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "id": "93958399-b1d0-432b-a78d-0619f707d9fd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "XGBClassifier(base_score=0.5, booster='gbtree', class_weight='balanced',\n",
      "              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,\n",
      "              criterion='gini', enable_categorical=False, gamma=0, gpu_id=-1,\n",
      "              importance_type=None, interaction_constraints='',\n",
      "              learning_rate=0.300000012, max_delta_step=0, max_depth=10,\n",
      "              max_features=1, min_child_weight=1, missing=nan,\n",
      "              monotone_constraints='()', n_estimators=40, n_jobs=4,\n",
      "              num_parallel_tree=1, predictor='auto', random_state=42,\n",
      "              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,\n",
      "              tree_method='exact', validate_parameters=1, ...)\n"
     ]
    }
   ],
   "source": [
    "print(grid_search_xgb.best_estimator_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 292,
   "id": "1c99e46f-aabb-4509-9654-692afa79d6db",
   "metadata": {},
   "outputs": [],
   "source": [
    "xgb_model1 = xgb.XGBClassifier(class_weight='balanced',learning_rate = 0.3002,\n",
    "                               random_state=42,refit='true', scoring='f1',verbose=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 293,
   "id": "6729eb76-2097-4188-b1c8-464a4b48a1a1",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages/xgboost/sklearn.py:1224: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].\n",
      "  warnings.warn(label_encoder_deprecation_msg, UserWarning)\n",
      "/home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages/xgboost/data.py:262: FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.\n",
      "  elif isinstance(data.columns, (pd.Int64Index, pd.RangeIndex)):\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[09:22:01] WARNING: ../src/learner.cc:576: \n",
      "Parameters: { \"class_weight\", \"refit\", \"scoring\", \"verbose\" } might not be used.\n",
      "\n",
      "  This could be a false alarm, with some parameters getting used by language bindings but\n",
      "  then being mistakenly passed down to XGBoost core, or some parameter actually being used\n",
      "  but getting flagged wrongly here. Please open an issue if you find any such cases.\n",
      "\n",
      "\n",
      "[09:22:01] WARNING: ../src/learner.cc:1115: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "XGBClassifier(base_score=0.5, booster='gbtree', class_weight='balanced',\n",
       "              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,\n",
       "              enable_categorical=False, gamma=0, gpu_id=-1,\n",
       "              importance_type=None, interaction_constraints='',\n",
       "              learning_rate=0.3002, max_delta_step=0, max_depth=6,\n",
       "              min_child_weight=1, missing=nan, monotone_constraints='()',\n",
       "              n_estimators=100, n_jobs=4, num_parallel_tree=1, predictor='auto',\n",
       "              random_state=42, refit='true', reg_alpha=0, reg_lambda=1,\n",
       "              scale_pos_weight=1, scoring='f1', subsample=1,\n",
       "              tree_method='exact', validate_parameters=1, ...)"
      ]
     },
     "execution_count": 293,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "xgb_model1.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 294,
   "id": "a0feca3d-54d0-491f-8d46-dddba824f621",
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_xgb1 = xgb_model1.predict(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f4747d39-9448-4839-a46d-22bb72d3b849",
   "metadata": {},
   "source": [
    "#### Model Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 295,
   "id": "2d60d8dc-6bf6-43de-8bbd-2a449fce41d8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy for tuned XGBoost: 0.830739299610895\n"
     ]
    }
   ],
   "source": [
    "print(\"Accuracy for tuned XGBoost:\", accuracy_score(y_test, pred_xgb1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 296,
   "id": "5d9fb343-60e2-4312-b6b2-3248424ca3da",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "F1 score for tuned XGBoost: 0.5583756345177665\n"
     ]
    }
   ],
   "source": [
    "print(\"F1 score for tuned XGBoost:\", f1_score(y_test, pred_xgb1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 306,
   "id": "042ef496-0bb7-4304-a288-8312af6a37ab",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix: \n",
      "          No Fire  Fire\n",
      "No Fire      372    30\n",
      "Fire          57    55\n"
     ]
    }
   ],
   "source": [
    "print(\"Confusion Matrix: \\n\", pd.DataFrame(confusion_matrix(y_test, pred_xgb1),columns=['No Fire','Fire'],index=['No Fire','Fire']))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "53e8cc49-5148-4e43-832a-fd3f1b14ea8f",
   "metadata": {},
   "source": [
    "#### With both grid search and \"trial and error\" I could not tune the hyperparameters to improve the xgboost baseline, so the model of choice will be random forest."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4e01a22b-bb4e-469f-b0d8-39b4318dd5fe",
   "metadata": {},
   "source": [
    "## Application"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "934eef97-61cc-466e-9dd1-ffb14a084031",
   "metadata": {
    "tags": []
   },
   "source": [
    "#### Now that the model has been built, it is time to apply it. An example application would be a simulation of a forest fire based on the chances of it occuring.\n",
    "\n",
    "#### We take our serialised model and use it in our application. The model has two functions of interest `predict` and `predict_proba`. In this case we will use the latter which outputs the probabilty value of whether a fire will occur or not.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "de4adb07-ea10-40cc-867b-c9c9bb7ac800",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the model\n",
    "import joblib\n",
    "rf_classifier = joblib.load(open(\"model_randomforest1.pkl\", \"rb\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "e7678a9d-263f-4200-9c99-912b25aa9480",
   "metadata": {},
   "outputs": [],
   "source": [
    "NDVI = 0.5\n",
    "LST = 15333\n",
    "BURNED_AREA = 0.4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "3407dedf-6b70-445d-8a7a-eee17793d7ee",
   "metadata": {},
   "outputs": [],
   "source": [
    "prob_fire = rf_classifier.predict_proba([[float(NDVI), float(LST),float(BURNED_AREA)]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "339f33af-8c7b-4a9b-8167-b3fe1d67ae6d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.567832776168178"
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fire_chance = prob_fire[0][1]\n",
    "fire_chance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "ad0e689e-bdf0-4d94-a90c-2e0aa871bdb8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: imageio in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (2.14.1)\n",
      "Requirement already satisfied: pillow>=8.3.2 in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from imageio) (9.0.0)\n",
      "Requirement already satisfied: numpy in /home/studio-lab-user/.conda/envs/default/lib/python3.9/site-packages (from imageio) (1.22.1)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install imageio"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2e5b7a1d-9703-4f35-89bd-a01727735332",
   "metadata": {},
   "source": [
    "### Simulation\n",
    "\n",
    "#### The simulation demonstrates that the higher the probabilty of a fire occuring, the bigger the expected spread.\n",
    "\n",
    "#### Reference: [How to simulate wildfires with python](https://medium.com/@tetraktyz/how-to-simulate-wildfires-with-python-6562e2eed266)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "0d71a0e3-8da5-4193-9b9d-899f1be1d245",
   "metadata": {},
   "outputs": [],
   "source": [
    "import imageio\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "c6cf075b-0a8a-4044-8af4-3c94e15384d5",
   "metadata": {},
   "outputs": [],
   "source": [
    "prob = fire_chance # probability of fire\n",
    "total_time = 300 # simulation time\n",
    "terrain_size = [100,100] # size of the simulation: 10000 cells"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "c64e38d9-5246-422b-839e-29285a74fdea",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<img src=\"video.gif\" width=\"500\"/>"
      ],
      "text/plain": [
       "<IPython.core.display.Image object>"
      ]
     },
     "execution_count": 83,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# states hold the state of each cell\n",
    "states = np.zeros((total_time,*terrain_size))\n",
    "# initialize states by creating random fuel and clear cells\n",
    "states[0] = np.random.choice([0,1],size=terrain_size,p=[1-prob,prob])\n",
    "# set the middle cell on fire!!!\n",
    "states[0,terrain_size[0]//2,terrain_size[1]//2] = 2\n",
    "\n",
    "for t in range(1,total_time):\n",
    "    # Make a copy of the original states\n",
    "    states[t] = states[t-1].copy()\n",
    "\n",
    "    for x in range(1,terrain_size[0]-1):\n",
    "        for y in range(1,terrain_size[1]-1):\n",
    "\n",
    "            if states[t-1,x,y] == 2: # It's on fire\n",
    "                states[t,x,y] = 0 # Put it out and clear it\n",
    "                \n",
    "                # If there's fuel surrounding it\n",
    "                # set it on fire!\n",
    "                if states[t-1,x+1,y] == 1: \n",
    "                    states[t,x+1,y] = 2\n",
    "                if states[t-1,x-1,y] == 1:\n",
    "                    states[t,x-1,y] = 2\n",
    "                if states[t-1,x,y+1] == 1:\n",
    "                    states[t,x,y+1] = 2\n",
    "                if states[t-1,x,y-1] == 1:\n",
    "                    states[t,x,y-1] = 2\n",
    "                    \n",
    "colored = np.zeros((total_time,*terrain_size,3),dtype=np.uint8)\n",
    "\n",
    "# Color\n",
    "for t in range(states.shape[0]):\n",
    "    for x in range(states[t].shape[0]):\n",
    "        for y in range(states[t].shape[1]):\n",
    "            value = states[t,x,y].copy()\n",
    "\n",
    "            if value == 0:\n",
    "                colored[t,x,y] = [139,69,19] # Clear\n",
    "            elif value == 1: \n",
    "                colored[t,x,y] = [0,255,0]   # Fuel\n",
    "            elif value == 2: \n",
    "                colored[t,x,y] = [255,0,0]   # Burning\n",
    "            \n",
    "# Crop\n",
    "cropped = colored[:200,1:terrain_size[0]-1,1:terrain_size[1]-1]\n",
    "\n",
    "imageio.mimsave('./video.gif', cropped)\n",
    "\n",
    "\n",
    "# define image_url variable\n",
    "image_url = \"video.gif\"\n",
    "\n",
    "# display image using url and specific width set at 500\n",
    "Image(url=image_url, width=500)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6f3151e6-f1f7-4794-b936-cc4bf91887ef",
   "metadata": {},
   "source": [
    "#### We have built a model to classify whether a fire will occur or not as well as the probabililties based on sensor readings. We have also applied the probabilities to simple simulation of the fire.\n",
    "\n",
    "#### A way to improve the concept is to add models for both the spread of the fire when it starts as well as the response. The model for the response can be an agent based on reinforcement learning to use autonomous drones to monitor and control fires as proposed in this paper by Ravi N. Haksar and Mac Schwager,\n",
    "#### [Distributed Deep Reinforcement Learning for Fighting Forest Fires with a Network of Aerial Robots*](https://web.stanford.edu/~schwager/MyPapers/HaksarSchwagerIROS18DRLFirefighting.pdf)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "923c6b78-c522-4e3d-b2a2-8d2e2427fe66",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "default:Python",
   "language": "python",
   "name": "conda-env-default-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
