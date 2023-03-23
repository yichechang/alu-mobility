import json
from typing import List, Dict

def main(fpath: str, confs: List[Dict]) -> None:
    conf = {**confs[0], **confs[1]}
    with open(fpath, 'w') as fp:
        json.dump(conf, fp, indent=4)

if __name__ == '__main__':
    main(snakemake.output[0],
         [
            snakemake.params['moviemeta'],
            snakemake.params['pivconfig'],
         ]
    )