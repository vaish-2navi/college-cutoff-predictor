
import joblib

model = joblib.load("model.pkl")

print("Input features:")
print(model.feature_names_in_)

print("\nPreprocessor details:")
print(model.named_steps["preprocessor"])

print("\nRegressor:")
print(model.named_steps["regressor"])
