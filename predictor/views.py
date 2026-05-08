from django.shortcuts import render
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "predictor", "mobile_price_model.pkl")

model = joblib.load(model_path)


def home(request):

    prediction = None
    price_label = ""

    accuracies = {
        "Logistic Regression": 96.5,
        "Decision Tree": 83.0,
        "Random Forest": 89.0,
        "KNN": 92.0,
        "SVM": 95.0,
        "Naive Bayes": 80.0,
    }

    best_algorithm = "Logistic Regression"

    if request.method == "POST":

        features = np.array([
            [
                int(request.POST["battery_power"]),
                int(request.POST["blue"]),
                float(request.POST["clock_speed"]),
                int(request.POST["dual_sim"]),
                int(request.POST["fc"]),
                int(request.POST["four_g"]),
                int(request.POST["int_memory"]),
                float(request.POST["m_dep"]),
                int(request.POST["mobile_wt"]),
                int(request.POST["n_cores"]),
                int(request.POST["pc"]),
                int(request.POST["px_height"]),
                int(request.POST["px_width"]),
                int(request.POST["ram"]),
                int(request.POST["sc_h"]),
                int(request.POST["sc_w"]),
                int(request.POST["talk_time"]),
                int(request.POST["three_g"]),
                int(request.POST["touch_screen"]),
                int(request.POST["wifi"]),
            ]
        ])

        prediction = model.predict(features)[0]

        if prediction == 0:
            price_label = "Low Cost Mobile"

        elif prediction == 1:
            price_label = "Medium Cost Mobile"

        elif prediction == 2:
            price_label = "High Cost Mobile"

        elif prediction == 3:
            price_label = "Very High Cost Mobile"

    return render(
        request,
        "index.html",
        {
            "prediction": prediction,
            "price_label": price_label,
            "accuracies": accuracies,
            "best_algorithm": best_algorithm,
        },
    )