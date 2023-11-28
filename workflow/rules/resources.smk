rule download_ilastik_models:
    params:
        url = lambda wildcards: config['resources']['ilastik'][wildcards.model_name]
    output:
        ilp="resources/ilastik/{model_name}.ilp"
    shell:
        """
        wget -O {output} {params.url}
        """

all_resources_input = []
all_resources_input.extend(
    expand(
        "resources/ilastik/{model_name}.ilp", 
        model_name=list(config['resources']['ilastik'].keys())
    )
)