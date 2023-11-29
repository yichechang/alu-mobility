rule download_ilastik_models:
    params:
        url = lambda wildcards: config['resources']['ilastik'][wildcards.model_name]
    output:
        ilp="resources/ilastik/{model_name}.ilp"
    shell:
        """
        wget -O {output} {params.url}
        """

rule download_example_data:
    params:
        url=config['resources']['example_data']['url'],
        outdir="resources/example_data",
        original_name=config['resources']['example_data']['name']
    output:
        directory("resources/example_data/")
    shell:
        """
        wget -P {params.outdir} --content-disposition {params.url:q}
        unzip {params.outdir}/{params.original_name} -x / -d {params.outdir}
        rm {params.outdir}/{params.original_name}
        """

all_resources_input = []
all_resources_input.extend(
    expand(
        "resources/ilastik/{model_name}.ilp", 
        model_name=list(config['resources']['ilastik'].keys())
    )
)

if config['resources']['example_data']['download']:
    all_resources_input.extend(["resources/example_data/"])