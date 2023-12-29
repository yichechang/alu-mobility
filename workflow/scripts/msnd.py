from msnd_utils import DataReader, MSNDPipeline


# get protocol definitions
protocol= snakemake.config['msnd']['protocols'][snakemake.wildcards.msnd_protocol]

# calculate msnd with data and
data = DataReader(snakemake.input, protocol['data'], snakemake.params['chnames']).read()
msnd = MSNDPipeline(data, protocol['preprocess'], protocol['process'])
df_stat, df_raw = msnd.calculate()

# save outputs
df_stat.to_csv(snakemake.output['stats'], index=False)
df_stat.to_csv(snakemake.output['indiv'], index=False)