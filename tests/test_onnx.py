import pytest
from src.onnx_export import export_model
from src.predictors import OnnxPredictor


@pytest.mark.parametrize(
    "text",
    ["heart coronary disease", "tumor chemotherapy", "words the model has never seen"],
)
def test_onnx_confidence_matches_sklearn(tiny_model, tmp_path, text):
    export_model(tiny_model, tmp_path)
    label, confidence = OnnxPredictor(tmp_path).predict(text)

    expected = tiny_model.predict_proba([text])[0].max()
    assert confidence == pytest.approx(expected, abs=1e-5)
    assert label in tiny_model.classes_
