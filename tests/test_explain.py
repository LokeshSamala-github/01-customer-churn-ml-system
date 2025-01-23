from churn_system.database import load_customer_features
from churn_system.explain import compute_shap_values, load_model, summarize_importance
from churn_system.features import build_inference_frame


def test_shap_values_shape_matches_input(trained_model):
    df = load_customer_features().sample(n=50, random_state=3)
    X = build_inference_frame(df)
    model = load_model()

    shap_values = compute_shap_values(model, X)
    assert shap_values.values.shape == X.shape


def test_feature_importance_is_sorted_descending(trained_model):
    df = load_customer_features().sample(n=50, random_state=3)
    X = build_inference_frame(df)
    model = load_model()
    shap_values = compute_shap_values(model, X)

    importance = summarize_importance(shap_values, X)
    values = importance["mean_abs_shap"].tolist()
    assert values == sorted(values, reverse=True)


def test_contract_type_is_a_top_driver(trained_model):
    """Sanity check tying the explainability output back to how the
    synthetic data was generated: contract type should dominate, since
    data/generate_data.py makes it by far the strongest hazard factor."""
    df = load_customer_features().sample(n=300, random_state=3)
    X = build_inference_frame(df)
    model = load_model()
    shap_values = compute_shap_values(model, X)
    importance = summarize_importance(shap_values, X)

    top_5 = set(importance.head(5)["feature"])
    assert any(f.startswith("contract_") for f in top_5)
