import json
import jsonschema
import tensorflow as tf
from tensorflow.keras.regularizers import l1, l2, l1_l2, OrthogonalRegularizer
from tensorflow.keras.initializers import glorot_uniform, he_normal, Zeros
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Add
from tensorflow.keras.layers import Conv2D, SimpleRNN, LSTM, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, sigmoid, tanh
from tensorflow.keras.layers import LeakyReLU, PReLU, ELU
from configuration_schemas import ffn_config_schema, short_ffn_config_schema 
def validate_json(data: dict, schema: dict) -> bool:
    try: 
        jsonschema.validate(instance=data, schema=schema)
        print("Validation passed!")
        return True
    except jsonschema.exceptions.ValidationError as err:
        print("Validation failed!")
        print(err.message)
        return False

def get_mappings():
    regularizer_mapping = {
        "2": l2,
        "1": l1,
        "12": l1_l2
    }    
    initializer_mapping = {
        "g": glorot_uniform,
        "h": he_normal,
        "z": Zeros
    }
    activation_mapping = {
        "r": relu,
        "s": sigmoid,
        "t": tanh,
        "l": LeakyReLU,
        "p": PReLU,
        "e": ELU
    }
    layer_type_mapping = {
        "d": Dense,
        "c": Conv2D,
        "s": SimpleRNN,
        "l": LSTM,
        "g": GRU
    }
    return regularizer_mapping, initializer_mapping, activation_mapping, layer_type_mapping
def perform_layer_operations(x: tf.Tensor, layer_params: dict, activation_mapping: dict) -> tf.Tensor:
    if 'dr' in layer_params:
        x = Dropout(layer_params['dr'])(x)
    if layer_params.get('bn', False):
        x = BatchNormalization()(x)
    if 'a' in layer_params:
        ActivationFunction = activation_mapping.get(layer_params['a'])
        if ActivationFunction:
            if isinstance(ActivationFunction, type):
                x = ActivationFunction()(x)
            else:
                x = ActivationFunction(x)
                #x = Activation(ActivationFunction)(x)        
    return x
def process_layer_params(layer_params: dict, layer_type_mapping: dict, regularizer_mapping: dict, initializer_mapping: dict, layer_type: str, i: int, num_layers: int) -> tuple:
    LayerType = layer_type_mapping.get(layer_type)
    if LayerType is None:
        raise ValueError("Invalid or no layer type specified.")
    modify_layer_regularizations(layer_params, regularizer_mapping)
    modify_layer_initializations(layer_params, initializer_mapping)
    # Dense layer requires 'units'.
    if layer_type == 'd':
        if 'u' not in layer_params:
            raise ValueError("For 'dense' layer type, 'units' parameter is required.")
        layer_params['units'] = layer_params.pop('u')    
    # Ensure RNN type layers in middle of model return sequences
    if layer_type in ['l', 'g', 's'] and i != num_layers - 1:
        layer_params['return_sequences'] = True
    return LayerType, layer_params
def strip_custom_params(layer_params: dict) -> dict:
    layer_params.pop('krl', None)   # kernel_regularizer_lambda
    layer_params.pop('brl', None)   # bias_regularizer_lambda
    layer_params.pop('dr', None)    # dropout_rate
    layer_params.pop('bn', None)    # batch_normalization
    layer_params.pop('a', None)     # activation
    layer_params.pop('r', None)     # residual
    return layer_params
def modify_layer_regularizations(layer_params: dict, regularizer_mapping: dict):
    reg_lambda_defaults = {'kr': 0.01, 'br': 0.01}  # Set default lambda values
    for reg_key, keras_reg_key in [('kr', 'kernel_regularizer'), ('br', 'bias_regularizer')]:
        if reg_key in layer_params:
            reg_lambda = float(layer_params.pop(f"{reg_key}l", reg_lambda_defaults[reg_key]))
            layer_params[keras_reg_key] = regularizer_mapping[layer_params.pop(reg_key)](reg_lambda)
def modify_layer_initializations(layer_params: dict, initializer_mapping: dict):
    for init_key, keras_init_key in [('ki', 'kernel_initializer'), ('bi', 'bias_initializer')]:
        if init_key in layer_params:
            Initializer = initializer_mapping.get(layer_params.pop(init_key))
            layer_params[keras_init_key] = Initializer()
def process_previous_layer_output(layer_params: dict, previous_layer_output: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
    if layer_params.get('r', False) and previous_layer_output is not None:
        try:
            x = Add()([x, previous_layer_output])
        except ValueError as ve:
            raise ValueError(f"Residual connection can't be applied due to incompatible shapes. Error: {ve}")
    return x
def build_flexible_model(input_tensor: tf.Tensor, config_json: str) -> tf.Tensor:
    regularizer_mapping, initializer_mapping, activation_mapping, layer_type_mapping = get_mappings()
    
    config = json.loads(config_json)

    # Call validate function
    if not validate_json(config, short_ffn_config_schema):
        raise ValueError("Invalid JSON configuration")
    x = input_tensor
    previous_layer_output = None
    # Process each layer according to config
    for i, layer_params in enumerate(config['l']): # layers
        layer_type = layer_params.pop('lt') # layer_type
        LayerType, layer_params = process_layer_params(layer_params, layer_type_mapping, regularizer_mapping, initializer_mapping, layer_type, i, len(config['l']))
        stripped_layer_params = strip_custom_params(layer_params.copy())  # Make a copy to keep the original dict intact
        # Validate input dimensions for specific layer types like Conv2D
        if layer_type == 'c' and len(x.shape) != 4: 
            raise ValueError(f"Input to 'Conv2D' should be a 4D tensor. Got {x.shape} tensor instead.")
        # Print layer parameters for debugging
        print("Configured layer with params:", stripped_layer_params)
        # Add layer to the model
        x = LayerType(**stripped_layer_params)(x)
        x = perform_layer_operations(x, layer_params, activation_mapping)
        x = process_previous_layer_output(layer_params, previous_layer_output, x)
        previous_layer_output = x
    # Add the output layer fixed to 100 neurons and softmax activation
    output_layer = Dense(100, activation='softmax')
    x = output_layer(x)
    # Construct and return the model
    model = Model(inputs=input_tensor, outputs=x)
    return model

json_string_config = """
{
  "l": [
    {"lt":"d","u":128,"kr":"2","br":"1","krl":0.01,"brl":0.02,"ki":"g","bi":"h","dr":0.3,"bn":true,"a":"r","r":false},
    {"lt":"d","u":64,"kr":"12","br":"2","krl":0.02,"brl":0.01,"ki":"h","bi":"g","dr":0.2,"bn":true,"a":"l","r":false}
  ]
}
"""
model = build_flexible_model(tf.keras.Input(shape=(10,)), json_string_config)
print("**************************Model Summary**************************")
print(model.summary())
print("**************************")
# print empty lines for better readability
print("")
print("**************************Model Layers**************************")
for layer in model.layers:
    print("")
    print("New Layer")
    print(layer.name)
    print(layer.get_config())
    print("")