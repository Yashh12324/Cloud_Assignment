import argparse
import mlflow
# ... other imports

def main(args):
    # Start MLflow logging
    mlflow.autolog(log_models=True, exclusive=True)

    df = load_data(args.data_path) # Use the argument
    X, y = prepare_xy(df)
    X_train, X_val, y_train, y_val = train_val_split(X, y, test_size=0.2)

    pipeline = build_pipeline()
    # ... (rest of your GridSearchCV code)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    metrics = evaluate_model(best, X_val, y_val)
    print("Validation metrics:", metrics)

    # The model is automatically saved by mlflow.autolog() to args.model_output_dir
    # No need for joblib.dump() or manual JSON saving

if _name_ == "_main_":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data")
    # MLflow autologger will use the model_output_dir implicitly
    # parser.add_argument("--model_output_dir", type=str, help="Path to save the model")
    args = parser.parse_args()
    main(args)
