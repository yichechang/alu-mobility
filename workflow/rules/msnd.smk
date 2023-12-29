def get_input_from_rule_outputs(input_type):
    # data type -> corresponding rule name and key to get output
    mapping = {
        "piv": {"rule": "piv", "key": 0},
        "mask_nuc": {"rule": "segment_nuclei_in_time", "key": 0},
        "mask_np": {"rule": "sn_convert_to_ometif", "key": 0},
    }
    rule_name = mapping[input_type]["rule"]
    output_key = mapping[input_type]["key"]
    rule_output = getattr(rules, rule_name).output
    return rule_output[output_key]

def get_input_files_for_protocol(wildcards):
    protocol_name = wildcards.msnd_protocol

    input_files = {}
    data_dict = config["msnd"]["protocols"][protocol_name]["data"]
    for data_name in data_dict:
        path = data_dict[data_name]["path"]
        # infer pattern from rule output if not specified
        if path is None:
            try:
                input_files[data_name] = (
                    get_input_from_rule_outputs(data_name)
                )
            except KeyError:
                raise ValueError(
                    f"Please specify input data pattern for "
                    f"'{data_name}' in config file. " 
                    f"Nothing provided and cannot infer file pattern "
                    f"from its name."
                )
        else:
            input_files[data_name] = path
    return input_files

rule msnd:
    input:
        unpack(get_input_files_for_protocol)
    output:
        stats="results/msnd/{protocol}/{ch}/{msnd_protocol}/{RoiUID}_stats.csv",
        indiv="results/msnd/{protocol}/{ch}/{msnd_protocol}/{RoiUID}_indiv.csv",
    params:
        chnames=ALL_CH,
    script:
        "../scripts/msnd.py"

all_msnd_input = [
    lambda w: expand(
        "results/msnd/{protocol}/{ch}/{msnd_protocol}/{RoiUID}_{outtype}.csv",
        protocol=ALL_PROTOCOLS,
        ch=config['msnd']['channel'],
        msnd_protocol=ALL_MSND_PROTOCOLS,
        outtype=['stats', 'indiv'],
        RoiUID=get_checkpoint_RoiUID(w),
    )
]