# abcpiv

PIV-based analysis for chromatin from A- or B-compartments.

---

Date started: 2023-02-02

---

### Dependencies

Dependencies are listed in [environments/abcd.yaml](environments/abcd.yaml) 
file and can be installed as a conda environment using either conda or
mamba. 

Currently, with mamba which is faster at solving, we need to first 
create an empty environment before we can install dependencies specified
in an yaml file. See the [issue and solution](https://github.com/mamba-org/mamba/issues/633#issuecomment-812272143).

```
mamba create -n abcd
mambe update -n abcd -f environments/abcd.yaml
```  
