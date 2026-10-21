"""JSON schema of the Component 3 -> Component 4 export (D7). Version 1.0.

Component 4 is NOT changed. It can read the JSON file (results/<run>/component4_export.json) or
call xaicomp.export.component4.recommendation(...) / window_explanation(...) in-process.
"""

ATTRIBUTION_ITEM = {
    "type": "object",
    "required": ["name", "attribution", "direction"],
    "properties": {
        "name": {"type": "string"},
        "attribution": {"type": "number"},
        "direction": {"enum": ["towards_target", "away_from_target", "neutral"]},
    },
}

WINDOW = {
    "type": "object",
    "required": ["context_id", "instance_id", "subject_id", "method", "prediction", "top_k", "trust", "in_sample"],
    "properties": {
        "context_id": {"type": "string"},
        "instance_id": {"type": "string"},
        "subject_id": {"type": "integer"},
        "t_start_s": {"type": "number"}, "t_end_s": {"type": "number"},
        "method": {"type": "string"},
        "prediction": {
            "type": "object", "required": ["target_class", "p_target", "predicted_class", "confidence"],
            "properties": {"target_class": {"type": "string"}, "p_target": {"type": "number", "minimum": 0, "maximum": 1},
                           "predicted_class": {"type": "string"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}}},
        "top_k": {
            "type": "object", "required": ["region", "band", "channel_x_band"],
            "properties": {lvl: {"type": "array", "items": ATTRIBUTION_ITEM}
                           for lvl in ("region", "band", "channel_x_band")}},
        "trust": {
            "type": "object", "required": ["level", "score", "components"],
            "properties": {"level": {"enum": ["high", "medium", "low"]}, "score": {"type": "number", "minimum": 0, "maximum": 1},
                           "components": {"type": "object"}, "notes": {"type": "array", "items": {"type": "string"}}}},
        "in_sample": {"type": "boolean"},
    },
}

CONTEXT = {
    "type": "object",
    "required": ["context_id", "run_type", "in_sample", "recommended_method", "p_rank1", "runner_up",
                 "practically_equivalent_to_runner_up", "equivalence", "criteria_used"],
    "properties": {
        "context_id": {"type": "string"},
        "title": {"type": "string"},
        "run_type": {"type": "string"},
        "in_sample": {"type": "boolean"},
        "recommended_method": {"type": "string"},
        "recommended_label": {"type": "string"},
        "p_rank1": {"type": "object", "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1}},
        "runner_up": {"type": "string"},
        "practically_equivalent_to_runner_up": {"type": "boolean"},
        "equivalence": {"type": "object", "required": ["p_recommended_better", "p_rope", "p_runner_up_better", "rope"],
                        "properties": {"p_recommended_better": {"type": ["number", "null"]}, "p_rope": {"type": ["number", "null"]},
                                       "p_runner_up_better": {"type": ["number", "null"]}, "rope": {"type": "number"},
                                       "measure": {"type": "string"}}},
        "criteria_used": {"type": "array", "items": {"type": "string"}},
        "plain_summary": {"type": "string"},
    },
}

EXPORT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Component 3 XAI recommendation export",
    "type": "object",
    "required": ["schema_version", "generated", "run", "contexts", "windows"],
    "properties": {
        "schema_version": {"const": "1.0"},
        "generated": {"type": "string"},
        "run": {"type": "object", "required": ["name", "run_type"],
                "properties": {"name": {"type": "string"}, "run_type": {"type": "string"}, "pilot": {"type": "boolean"}}},
        "contexts": {"type": "array", "items": CONTEXT},
        "windows": {"type": "array", "items": WINDOW},
    },
}


def validate(doc: dict) -> None:
    import jsonschema
    jsonschema.validate(doc, EXPORT_SCHEMA)
