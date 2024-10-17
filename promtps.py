system_prompt = """
    You are a synthetic data science developer focused on optimizing a Feed-Forward Network (FFN) to achieve over 90 percent accuracy in predicting outcomes. 
    You are a part of the automated pipeline which configures and trains the model, predicts and evaluates the results.
    The FFN represents itself an input layer with several amount of features, x hidden layers, and an output layer with 100 classes (softmax activation).
    To configure the FFN, a function requires JSON input detailing key model parameters and parameters for hidden layers and returns the corresponding FFN model. 
    You have access to records of previous configurations and their performances but must avoid any repetitions. Repetion is considered only when all parameters are the same.
    To efficiently manage the token count used in the Large Language Model (LLM), the configuration process is divided based on the number of hidden layers. 
    For each iteration, you focus on a specific number of layers before shifting to configurations with different layer counts. 
    This method ensures minimal repetition of parameter details per LLM interaction.
    Your efforts going forward involve tweaking these configurations, ensuring you explore new configurations within and beyond the current number of layers to elevate model performance effectively.
    The model configuration parameters are defined in a JSON-schema below:
    JSON: {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "cfg": { // model configuration 
            "type": "object",
            "properties": {
                "lm": {"type": "number"},  // lambda
                "bs": {"type": "number"},  // batch_size
                "ep": {"type": "number"},  // epochs
                "lr": {"type": "number"},   // learning_rate
                "lf": {"type": "number"},   // learning_rate_factor
                "lp": {"type": "number"},   // learning_rate_patience
                "md": {"type": "number"},   // min_delta
                "cd": {"type": "number"},   // cooldown
                "mlr": {"type": "number"},  // min_learning_rate                                
                "esp": {"type": "number"}  // early_stopping_patience
            },
            "required": ["lm", "bs", "ep", "lr", "lf", "lp", "md", "cd", "mlr", "esp"]
        },            
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

def get_first_request_prompt(pipeline_data: dict) -> str:
    input_layers_ct = 50
    output_layers_ct = 100
    return f"""
    Generate the first configuration for the Feed-Forward Network (FFN).
    Start with the one hidden layer configuration.
    The input layer has {input_layers_ct} units and the output layer has {output_layers_ct} classes (softmax activation).
    Generate configuration for the model and hidden layers only, as input and output layers are already defined. 
    No comments are allowed as an automated function will use the output. Ensure the response is a valid JSON. Aviod any formatting make JSON as a single line.
    """
def get_previous_layers_perf(pipeline_data: dict) -> str:
    prev_layers_perf = ""
    if "hidden_layers_configs" in pipeline_data:
        for layer in pipeline_data["hidden_layers_configs"]:
            if layer["is_completed"]:
                prev_layers_perf += f"\n{layer['hidden_layers_ct']} hidden layers with MAX accuracy: {layer['MAX_accuracy']} "
    return prev_layers_perf

def get_new_layers_request_prompt(pipeline_data: dict, hidden_layers_ct: int) -> str:
    input_layers_ct = 50
    output_layers_ct = 100

    prev_layers_perf = get_previous_layers_perf(pipeline_data)

    return f"""
    Generate the first configuration for the Feed-Forward Network (FFN) which has {hidden_layers_ct} hidden layers.
    The input layer has {input_layers_ct} units and the output layer has {output_layers_ct} classes (softmax activation).    
    You have already used the following hidden layers quantities with the MAX corresponding accuracy: {prev_layers_perf}
    Generate configuration for the model and hidden layers only, as input and output layers are already defined.
    No comments are allowed as an automated function will use the output. Ensure the response is a valid JSON. Aviod any formatting make JSON as a single line.
    """
    
def get_curr_layer_perf(pipeline_data: dict) -> str:
    curr_layer_perf = ""
    hidden_layers_ct = pipeline_data["current_hidden_layers_ct"]

    if "hidden_layers_configs" in pipeline_data:
        for layer in pipeline_data["hidden_layers_configs"]:
            if layer["hidden_layers_ct"] == hidden_layers_ct:
                for config in layer["configurations"]:
                    curr_layer_perf += f"\n configuration: {config['configuration']} - accuracy: {config['accuracy']}"
    
    if curr_layer_perf == "":
        curr_layer_perf = "None"

    return curr_layer_perf

def get_next_layers_request_prompt(pipeline_data: dict) -> str:
    input_layers_ct = 50
    output_layers_ct = 100
    hidden_layers_ct = pipeline_data["current_hidden_layers_ct"]
    prev_layers_perf = get_previous_layers_perf(pipeline_data)
    curr_layer_perf = get_curr_layer_perf(pipeline_data)

    return f"""
    Generate the next configuration for the Feed-Forward Network (FFN) which has {hidden_layers_ct} hidden layers.
    The input layer has {input_layers_ct} units and the output layer has {output_layers_ct} classes (softmax activation).
    You have already used the following hidden layers quantities with the MAX corresponding accuracy: {prev_layers_perf}
    Within the current number of hidden layers, you have already used the following configurations with the corresponding accuracy: {curr_layer_perf}
    Generate configuration for the model and hidden layers only, as input and output layers are already defined.
    Generate a new configuration only and avoid repetitions.
    No comments are allowed as an automated function will use the output.
    Ensure the response is a valid JSON. Aviod any formatting make JSON as a single line.
    You also have the permission to suggest switching to another number of hidden layers. In such case, you should respond "SWITCH TO X HIDDEN LAYERS"
    Where X is the number of hidden layers you want to switch to.
    You can switch only to a number of hidden layers you have not used before (higher than the current number of hidden layers).    
    """


additional_prompt = """    
    Generate the next configuration for the Feed-Forward Network (FFN).
    Generate a new configuration only. Generate configuration for the model and hidden layers only, as input and output layers are already defined. 
    No comments are allowed as an automated function will use the output.  
    You also have the permission to suggest switching to another number of hidden layers. In such case, you should respond "SWITCH TO X HIDDEN LAYERS"
    It is important to note that if you switch to a different number of hidden layers, you will never be able to return to the number of hidden layers you used before.
"""

def get_input_layers_ct() -> int:
    return 50

def generate_LLM_prompt(pipeline_data: dict, need_new_layer: int) -> str:    
    # need_new_layer is 0 if no new layer is needed, and equals to the number of new layers needed otherwise
    prompt = None

    if "current_hidden_layers_ct" not in pipeline_data:
        prompt = get_first_request_prompt(pipeline_data)        
        return prompt
        
    if need_new_layer > 0:
        prompt = get_new_layers_request_prompt(pipeline_data, need_new_layer)
        return prompt 
    
    prompt = get_next_layers_request_prompt(pipeline_data)
    return prompt
    
# #Testing dictionary to check the new pipline promprt generation
# pipeline_dict_test_new_pipeline = {
#     "pipeline_id": "1234",
#     "status": 1
#     }
# #Testing dictionary to check the new hidden layers count prompt generation
# pipeline_dict_test_new_hidden_layers = {
#     "pipeline_id": "1234",
#     "status": 1,
#     "current_hidden_layers_ct": 1,
#     "hidden_layers_configs": [
#         {
#             "hidden_layers_ct": 1,
#             "is_completed": True,
#             "MAX_accuracy": 0.01
#         },
#         {
#             "hidden_layers_ct": 2,
#             "is_completed": True,
#             "MAX_accuracy": 0.02
#         },
#         {
#             "hidden_layers_ct": 3,
#             "is_completed": True,
#             "MAX_accuracy": 0.03
#         }
#     ]
#     }

# #Testing dictionary to check the next configuration for the same hidden layers count prompt generation
# pipeline_dict_test_next_hidden_layers = {
#     "pipeline_id": "1234",
#     "status": 1,
#     "current_hidden_layers_ct": 1,
#     "hidden_layers_configs": [
#         {
#             "hidden_layers_ct": 1,
#             "is_completed": False,
#             "MAX_accuracy": 0.9,
#             "configurations": [
#                 {
#                     "configuration": "{'l': [{'lt': 'd', 'u': 100, 'kr': '2', 'br': '2', 'krl': 0.001, 'brl': 0.001, 'ki': 'g', 'bi': 'g', 'dr': 0.2, 'bn': true, 'a': 'r', 'r': false}]}",
#                     "accuracy": 0.5
#                 }
#             ]
#         },
#                 {
#             "hidden_layers_ct": 1,
#             "is_completed": False,
#             "MAX_accuracy": 0.9,
#             "configurations": [
#                 {
#                     "configuration": "{'l': [{'lt': 'd', 'u': 100, 'kr': '2', 'br': '2', 'krl': 0.001, 'brl': 0.001, 'ki': 'g', 'bi': 'g', 'dr': 0.2, 'bn': true, 'a': 'r', 'r': false}]}",
#                     "accuracy": 0.5
#                 }
#             ]
#         },
#                 {
#             "hidden_layers_ct": 1,
#             "is_completed": False,
#             "MAX_accuracy": 0.9,
#             "configurations": [
#                 {
#                     "configuration": "{'l': [{'lt': 'd', 'u': 100, 'kr': '2', 'br': '2', 'krl': 0.001, 'brl': 0.001, 'ki': 'g', 'bi': 'g', 'dr': 0.2, 'bn': true, 'a': 'r', 'r': false}]}",
#                     "accuracy": 0.5
#                 }
#             ]
#         }
#     ]
#     }

# #print(generate_LLM_prompt(pipeline_dict_test_new_pipeline, 1))
# print(generate_LLM_prompt(pipeline_dict_test_new_hidden_layers, 2))
# #print(generate_LLM_prompt(pipeline_dict_test_next_hidden_layers, 0))





    

    
#     # prompt = prompt + " " + load_previous_model_configurations()
# prompt = ""
# prompt = prompt + " " + additional_prompt
    
