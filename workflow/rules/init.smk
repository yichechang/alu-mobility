rule config_template:
    output: "config.yaml"
    run:
        from pathlib import Path
        import shutil

        config_path = Path(workflow.basedir) / "../config/config.yaml"
        shutil.copyfile(config_path, output[0])

all_init_input = [
    "config.yaml",
]