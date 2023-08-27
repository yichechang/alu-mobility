rule fit_msnd_line:
    input:
        msnd="results/{msnd_type}/{protocol}/{ch}/{msnd_protocol}/{RoiUID}_stats.csv",
    output:
        csv="results/fit_msnd/{msnd_type}/{protocol}/{ch}/{msnd_protocol}/{RoiUID}_fitparams.csv"
    run:
        import numpy as np
        import pandas as pd
        from scipy.optimize import curve_fit
        
        def powerlaw(x, A, B):
            return B * (x ** A)

        def _curve_fit(df):
            popt, pcov = curve_fit(
                f=powerlaw,
                xdata=df['lag_s'].values,
                ydata=df['mean'].values,
                p0=[0.25, 10 ** (-2)]
            )
            fit = pd.Series(
                {
                    'alpha': popt[0], 
                    'Dapp': popt[1]/4,
                    'alpha_cov': np.diag(pcov)[0],
                    'Dapp_cov': np.diag(pcov)[1],
                }
            )
            return fit
        
        min_size = config['msnd_alpha']['min_size'][wildcards.msnd_protocol]
        df = pd.read_csv(input.msnd).dropna().query('size >= @min_size')
        df['log_lag_s'] = np.log10(df['lag_s'])
        df['log_mean'] = np.log10(df['mean'])

        by = config['msnd_alpha']['groupby'][wildcards.msnd_protocol]
        if by is not None:
            df['npoints'] = df.groupby(by)['mean'].transform('count')
            min_npoints = config['msnd_alpha']['min_npoints'][wildcards.msnd_protocol]
            fit = df.query('npoints > @min_npoints').groupby(by).apply(_curve_fit).reset_index()
        else:
            fit = _curve_fit(df).to_frame().T
 
        fit['RoiUID'] = wildcards.RoiUID
        fit.to_csv(output.csv, index=False)

rule instantaneous_alphas:
    input:
        msnd="results/{msnd_type}/{protocol}/{ch}/{msnd_protocol}/{RoiUID}_indiv.csv",
    output:
        csv="results/insta_alpha/{msnd_type}/{protocol}/{ch}/{msnd_protocol}/{RoiUID}_alphas.csv"
    run:
        import numpy as np
        import pandas as pd
        def _calc_slope_in_one_group(df: pd.DataFrame):
            lagdiff = df['log_lag_s'].diff()
            lag = (lagdiff / 2 + df['log_lag_s'])
            slope = (df['log_msnd'].diff() / lagdiff)
            df_slope = pd.DataFrame({'log_lag_s': lag, 'alpha': slope})

            return df_slope.dropna().set_index('log_lag_s')
        
        def compute_instantaneous_alpha(df: pd.DataFrame):
            # convert to log-log space
            df['log_lag_s'] = np.log10(df['lag_s'])
            df['log_msnd'] = np.log10(df['msnd'])

            return df.groupby('T_s').apply(_calc_slope_in_one_group)
        
        df = pd.read_csv(input.msnd).dropna()

        by = config['msnd_alpha']['groupby'][wildcards.msnd_protocol]
        if by is not None:
            alphas = df.groupby(by).apply(compute_instantaneous_alpha).reset_index()
        else:
            alphas = compute_instantaneous_alpha(df).reset_index()
        
        alphas['RoiUID'] = wildcards.RoiUID
        alphas.to_csv(output.csv, index=False)


all_msnd_alpha_input = [
    lambda w: expand(
        "results/fit_msnd/{msnd_type}/{protocol}/{ch}/{msnd_protocol}/{RoiUID}_fitparams.csv",
        msnd_type=['msnd', 'msnf'],
        protocol=ALL_PROTOCOLS,
        ch=config['msnd_alpha']['channel'],
        msnd_protocol=['normal', 'eachlevel'],
        RoiUID=get_checkpoint_RoiUID(w)
    ),
    lambda w: expand(
        "results/insta_alpha/{msnd_type}/{protocol}/{ch}/{msnd_protocol}/{RoiUID}_alphas.csv",
        msnd_type=['msnd', 'msnf'],
        protocol=ALL_PROTOCOLS,
        ch=config['msnd_alpha']['channel'],
        msnd_protocol=['normal', 'eachlevel'],
        RoiUID=get_checkpoint_RoiUID(w)
    )
]