system_prompt = """
    You are a synthetic data science developer focused on optimizing a Feed-Forward Network (FFN) to achieve over 90 percent accuracy in predicting outcomes. 
    You are a part of the automated pipeline which configures and trains the model, predicts and evaluates the results. 
    To configure the FFN, a function requires JSON input detailing model parameters for hidden layers and returns the corresponding FFN model. 
    You have access to records of previous configurations and their performances but must avoid any repetitions.
    To efficiently manage the token count used in the Large Language Model (LLM), the configuration process is divided based on the number of hidden layers. 
    For each iteration, you focus on a specific number of layers before shifting to configurations with different layer counts. 
    This method ensures minimal repetition of parameter details per LLM interaction.
    Your efforts going forward involve tweaking these configurations, ensuring you explore new configurations within and beyond the current number of layers to elevate model performance effectively.
    The model configuration parameters are defined in a JSON-schema below:
    JSON: {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "l": {  // Layer configurations
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
            "lt": {"type": "string", "enum": ["d", "c", "s", "l", "g"]},  // Layer types: dense, conv2d, simple_rnn, lstm, gru
            "u": {"type": ["number", "null"]},                           // Units
            "kr": {"type": ["string", "null"], "enum": ["2", "1", "12"]},// kernel regularizer: l2, l1, l1_l2
            "br": {"type": ["string", "null"], "enum": ["2", "1", "12"]}, // bias initializer: l2, l1, l1_l2
            "krl": {"type": ["number", "null"]},                         // Kernel regularization lambdas
            "brl": {"type": ["number", "null"]},                         // Bias regularization lambdas
            "ki": {"type": ["string", "null"], "enum": ["g", "h", "z"]}, // Kernel Initializers glorot_uniform, he_normal, zeros
            "bi": {"type": ["string", "null"], "enum": ["g", "h", "z"]}, // Bias glorot_uniform, he_normal, zeros
            "dr": {"type": ["number", "null"]},                          // Dropout rate
            "bn": {"type": ["boolean", "null"]},                        // Batch normalization
            "a": {"type": ["string", "null"], "enum": ["r", "s", "t", "l", "p", "e"]},  // Activations: relu, sigmoid, tanh, leaky_relu, prelu, elu
            "r": {"type": ["boolean", "null"]}                          // Residual connections
            },
            "required": ["lt"]
            }
            }
        },
        "required": ["l"],
        "additionalProperties": False
    }
    """

def get_first_request_prompt(input_layers_ct: int, output_layers_ct: int) -> str:
    return f"""
    Generate the first configuration for the Feed-Forward Network (FFN).
    Start with the one hidden layer configuration.
    The input layer has {input_layers_ct} units and the output layer has {output_layers_ct} units.
    Generate configuration for the hidden layers only, as input and output layers are already defined. 
    No comments are allowed as an automated function will use the output. Ensure the response is a valid JSON.
    """

def get_new_layers_request_prompt(input_layers_ct: int, output_layers_ct: int, hidden_layers_ct: int, prev_layers_perf: str) -> str:
    return f"""
    Generate the first configuration for the Feed-Forward Network (FFN) which has {hidden_layers_ct} hidden layers.
    The input layer has {input_layers_ct} units and the output layer has {output_layers_ct} units.    
    You have already used the following hidden layers quantities with the MAX corresponding accuracy: {prev_layers_perf}
    Generate configuration for the hidden layers only, as input and output layers are already defined.
    No comments are allowed as an automated function will use the output. Ensure the response is a valid JSON.
    """
    
def get_next_layers_request_prompt(input_layers_ct: int, output_layers_ct: int, hidden_layers_ct: int, prev_layers_perf: str, curr_layer_perf) -> str:
    return f"""
    Generate the next configuration for the Feed-Forward Network (FFN) which has {hidden_layers_ct} hidden layers.
    The input layer has {input_layers_ct} units and the output layer has {output_layers_ct} units.    
    You have already used the following hidden layers quantities with the MAX corresponding accuracy: {prev_layers_perf}
    Within the current number of hidden layers, you have already used the following configurations with the corresponding accuracy: {curr_layer_perf}
    Generate configuration for the hidden layers only, as input and output layers are already defined.
    Generate a new configuration only and avoid repetitions.
    No comments are allowed as an automated function will use the output.
    Ensure the response is a valid JSON.
    You also have the permission to suggest switching to another number of hidden layers. In such case, you should respond "SWITCH TO X HIDDEN LAYERS"
    Where X is the number of hidden layers you want to switch to.
    You can switch only to a number of hidden layers you have not used before (higher than the current number of hidden layers).    
    """


additional_prompt = """    
    Generate the next configuration for the Feed-Forward Network (FFN).
    Generate a new configuration only. Generate configuration for the hidden layers only, as input and output layers are already defined. 
    No comments are allowed as an automated function will use the output.  
    You also have the permission to suggest switching to another number of hidden layers. In such case, you should respond "SWITCH TO X HIDDEN LAYERS"
    It is important to note that if you switch to a different number of hidden layers, you will never be able to return to the number of hidden layers you used before.
"""