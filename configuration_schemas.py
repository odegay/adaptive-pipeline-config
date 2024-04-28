ffn_config_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "layers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "layer_type": {
                        "type": "string",
                        "enum": ["dense", "conv2d", "simple_rnn", "lstm", "gru"]
                    },
                    "units": {
                        "type": ["number", "null"]                        
                    },
                    "kernel_regularizer": {
                        "type": ["string", "null"],
                        "enum": ["l2", "l1", "l1_l2"]
                    },
                    "bias_regularizer": {
                        "type": ["string", "null"],
                        "enum": ["l2", "l1", "l1_l2"]
                    },
                    "kernel_regularizer_lambda": {
                        "type": ["number", "null"]
                    },
                    "bias_regularizer_lambda": {
                        "type": ["number", "null"]
                    },
                    "kernel_initializer": {
                        "type": ["string", "null"],
                        "enum": ["glorot_uniform", "he_normal", "zeros"]
                    },
                    "bias_initializer": {
                        "type": ["string", "null"],
                        "enum": ["glorot_uniform", "he_normal", "zeros"]
                    },
                    "dropout_rate": {
                        "type": ["number", "null"]
                    },
                    "batch_normalization": {
                        "type": ["boolean", "null"]
                    },
                    "activation": {
                        "type": ["string", "null"],
                        "enum": ["relu", "sigmoid", "tanh", "leaky_relu", "prelu", "elu"]
                    },
                    "residual": {
                        "type": ["boolean", "null"]
                    }
                },
                "additionalProperties": {
                    "type": ["number", "string", "boolean"]
                },
                "required": ["layer_type"]
            }
        }
    },
    "required": ["layers"],
    "additionalProperties": False
}

short_ffn_config_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "l": {  # layers
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "lt": {  # layer_type
                        "type": "string",
                        "enum": ["d", "c", "s", "l", "g"]  # dense, conv2d, simple_rnn, lstm, gru
                    },
                    "u": {  # units
                        "type": ["number", "null"]
                    },
                    "kr": {  # kernel_regularizer
                        "type": ["string", "null"],
                        "enum": ["2", "1", "12"]  # l2, l1, l1_l2
                    },
                    "br": {  # bias_regularizer
                        "type": ["string", "null"],
                        "enum": ["2", "1", "12"]  # l2, l1, l1_l2
                    },
                    "krl": {  # kernel_regularizer_lambda
                        "type": ["number", "null"]
                    },
                    "brl": {  # bias_regularizer_lambda
                        "type": ["number", "null"]
                    },
                    "ki": {  # kernel_initializer
                        "type": ["string", "null"],
                        "enum": ["g", "h", "z"]  # glorot_uniform, he_normal, zeros
                    },
                    "bi": {  # bias_initializer
                        "type": ["string", "null"],
                        "enum": ["g", "h", "z"]  # glorot_uniform, he_normal, zeros
                    },
                    "dr": {  # dropout_rate
                        "type": ["number", "null"]
                    },
                    "bn": {  # batch_normalization
                        "type": ["boolean", "null"]
                    },
                    "a": {  # activation
                        "type": ["string", "null"],
                        "enum": ["r", "s", "t", "l", "p", "e"]  # relu, sigmoid, tanh, leaky_relu, prelu, elu
                    },
                    "r": {  # residual
                        "type": ["boolean", "null"]
                    }
                },
                "additionalProperties": {
                    "type": ["number", "string", "boolean"]
                },
                "required": ["lt"]
            }
        }
    },
    "required": ["l"],  # layers
    "additionalProperties": False
}