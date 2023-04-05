rule download_ilastik_models:
    output:
        expand("resources/ilastik/{model_name}.ilp",
               model_name=config['predict_nucleoli']['model_name'])
    params:
        url=(config['resources']
                   ['classifiers']
                   [config['predict_nucleoli']['model_name']]
                   ['url'])
    shell:
        """
        wget -O {output} {params.url}
        """

all_resources_input = []
all_resources_input.extend(rules.download_ilastik_models.output)